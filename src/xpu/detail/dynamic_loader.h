#ifndef XPU_DETAIL_DYNAMIC_LOADER_H
#define XPU_DETAIL_DYNAMIC_LOADER_H

#include "../common.h"
#include "../defines.h"
#include "../driver/cpu/this_thread.h"
#include "log.h"
#include "macros.h"
#include "type_info.h"

#if XPU_IS_HIP
#include <hip/hip_runtime_api.h>
#endif

#include <dlfcn.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace xpu {
namespace detail {

template<typename I, typename T>
struct action {
    using image = I;
    using tag = T;
};

template<typename...>
struct action_interface {};

template<typename Tag, typename... Args>
struct action_interface<Tag, void(*)(Args...)> {
    using type = int(*)(Args...);
};

template<typename S, typename... Args>
struct action_interface<kernel_tag, void(*)(S, Args...)> {
    using type = int(*)(float *, grid, Args...);
};

template<typename... Args>
struct action_interface<constant_tag, int(*)(Args...)> {
    using type = int(*)(Args...);
};

struct delete_me {};

template<typename T, typename A>
struct action_interface<T, A> : action_interface<T, decltype(&A::impl)> {};

template<typename A>
struct action_interface<kernel_tag, A> : action_interface<kernel_tag, decltype(&A::template impl<delete_me>)> {};

template<typename A>
using action_interface_t = typename action_interface<typename A::tag, A>::type;

using symbol_table = std::vector<std::pair<std::string, void *>>;

template <typename I>
class image_context {

public:
    static const char *file_name;
    static std::unordered_map<std::string, size_t> &ids();

    static image_context<I> *instance();

    image_context() {
        name = type_name<I>();
    }

    symbol_table &get_symbols() { return symbols; }
    std::string get_name() const { return name; }

    template<typename A>
    void add_symbol(void *symbol) {
        auto it = ids().find(type_name<A>());
        if (it == ids().end()) {
            ids()[type_name<A>()] = type_id<A, typename A::image>::get();
        }
        size_t id = ids()[type_name<A>()];
        if (symbols.size() <= id) {
            symbols.resize(id+1);
        }
        symbols.at(id) = {type_name<A>(), symbol};
    }

private:
    symbol_table symbols;

    std::string name;

};

template<typename I>
class image {

private:
    void *handle = nullptr;

    image_context<I> *context = nullptr;

public:
    image() {
        context = image_context<I>::instance();
    }

    image(const char *name) {
#if defined __APPLE__
  handle = dlopen(name, RTLD_LAZY);
#elif defined __linux__
  handle = dlopen(name, RTLD_LAZY | RTLD_DEEPBIND);
#endif
        if (handle == nullptr) {
            XPU_LOG("Error opening '%s: %s", name, dlerror());
        }
        assert(handle != nullptr);
        auto *get_context = reinterpret_cast<image_context<I> *(*)()>(dlsym(handle, "xpu_detail_get_context"));
        assert(get_context != nullptr);
        context = get_context();
        assert(context->get_name() == type_name<I>());
    }

    ~image() {
        if (handle != nullptr) {
            dlclose(handle);
        }
    }

    image(const image<I> &) = delete;

    image(image<I> &&other) {
        handle = other.handle;
        context = other.context;

        other.handle = nullptr;
        other.context = nullptr;
    }

    template<typename F, typename... Args>
    typename std::enable_if<is_function<I, F>::value>::type call(Args&&... args) {
        return call_action<F>(args...);
    }

    template<typename F>
    typename std::enable_if<is_constant<I, F>::value, int>::type set(const typename F::data_t &val) {
        return call_action<F>(val);
    }

    template<typename K, typename... Args>
    typename std::enable_if<is_kernel<I, K>::value, int>::type run_kernel(float *ms, grid g, Args&&... args) {
        return call_action<K>(ms, g, args...);
    }

    void dump_symbols() {
        for (auto it : context->get_symbols()) {
            XPU_LOG("%s: %p", it.first.c_str(), it.second);
        }
    }

private:
    template<typename F, typename... Args>
    int call_action(Args... args) {
        auto *symbols = &context->get_symbols();
        size_t id = type_id<F, typename F::image>::get();
        if (id >= symbols->size()) {
            dump_symbols();
        }
        assert(id < symbols->size());
        auto symbol = symbols->at(id);
        assert(symbol.first == type_name<F>());
        auto *fn = reinterpret_cast<action_interface_t<F>>(symbol.second);
        return fn(args...);
    }

};

template<typename...>
struct action_runner {};

template<typename Tag, typename F, typename... Args>
struct action_runner<Tag, F, void(*)(Args...)> {

    using my_type = action_runner<Tag, F, void(*)(Args...)>;

    static int call(Args... args) {
        F::impl(args...);
        return 0;
    }
};

template<typename F, typename... Args>
struct action_runner<constant_tag, F, int(*)(Args...)> {

    using my_type = action_runner<constant_tag, F, int(*)(Args...)>;

    static int call(Args... args) {
        return F::impl(args...);
    }
};

template<typename F>
struct action_runner<F> : action_runner<typename F::tag, F, decltype(&F::impl)> {};

#if XPU_IS_HIP_CUDA

#define SAFE_CALL(call) if (int err = call; err != 0) return err
#define ON_ERROR_GOTO(errvar, call, label) errvar = call; \
    if (errvar != 0) goto label

template<typename F, typename S, typename... Args>
__global__ void kernel_entry(Args... args) {
    __shared__ S smem;
    F::impl(smem, args...);
}

#endif

#if XPU_IS_CUDA

template<typename K, typename S, typename... Args>
struct action_runner<kernel_tag, K, S, void(*)(S &, Args...)> {

    using my_type = action_runner<kernel_tag, K, S, void(*)(S &, Args...)>;

    static int call(float *ms, grid g, Args... args) {
        int bsize = block_size<K>::value;

        if (g.blocks.x == -1) {
            g.blocks.x = (g.threads.x + bsize - 1) / bsize;
        }

        bool measure_time = (ms != nullptr);
        cudaEvent_t start, end;
        int err;

        if (measure_time) {
            SAFE_CALL(cudaEventCreate(&start));
            ON_ERROR_GOTO(err, cudaEventCreate(&end), cleanup_start_event);
        }

        XPU_LOG("Calling kernel '%s' [block_dim = (%d, %d, %d), grid_dim = (%d, %d, %d)] with CUDA driver.", type_name<K>(), bsize, 0, 0, g.blocks.x, 0, 0);

        if (measure_time) {
            ON_ERROR_GOTO(err, cudaEventRecord(start), cleanup_events);
        }

        kernel_entry<K, S, Args...><<<g.blocks.x, bsize>>>(args...);

        if (measure_time) {
            ON_ERROR_GOTO(err, cudaEventRecord(end), cleanup_events);
        }
        SAFE_CALL(cudaDeviceSynchronize());

        if (measure_time) {
            ON_ERROR_GOTO(err, cudaEventSynchronize(end), cleanup_events);
            ON_ERROR_GOTO(err, cudaEventElapsedTime(ms, start, end), cleanup_events);
            XPU_LOG("Kernel '%s' took %f ms", type_name<K>(), *ms);
        }

    cleanup_events:
        if (measure_time) {
            SAFE_CALL(cudaEventDestroy(end));
        }
    cleanup_start_event:
        if (measure_time) {
            SAFE_CALL(cudaEventDestroy(start));
        }

        return err;
    }

};

#elif XPU_IS_HIP

template<typename K, typename S, typename... Args>
struct action_runner<kernel_tag, K, S, void(*)(S &, Args...)> {

    using my_type = action_runner<kernel_tag, K, S, void(*)(S &, Args...)>;

    static int call(float *ms, grid g, Args... args) {
        int bsize = block_size<K>::value;

        if (g.blocks.x == -1) {
            g.blocks.x = (g.threads.x + bsize - 1) / bsize;
        }

        bool measure_time = (ms != nullptr);
        hipEvent_t start, end;
        int err = 0;

        if (measure_time) {
            SAFE_CALL(hipEventCreate(&start));
            ON_ERROR_GOTO(err, hipEventCreate(&end), cleanup_start_event);
        }

        XPU_LOG("Calling kernel '%s' [block_dim = (%d, %d, %d), grid_dim = (%d, %d, %d)] with HIP driver.", type_name<K>(), bsize, 0, 0, g.blocks.x, 0, 0);
        if (measure_time) {
            ON_ERROR_GOTO(err, hipEventRecord(start), cleanup_events);
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_entry<K, S, Args...>), dim3(g.blocks.x), dim3(bsize), 0, 0, args...);
        if (measure_time) {
            ON_ERROR_GOTO(err, hipEventRecord(end), cleanup_events);
        }
        SAFE_CALL(hipDeviceSynchronize());

        if (measure_time) {
            ON_ERROR_GOTO(err, hipEventSynchronize(end), cleanup_events);
            ON_ERROR_GOTO(err, hipEventElapsedTime(ms, start, end), cleanup_events);
            XPU_LOG("Kernel '%s' took %f ms", type_name<K>(), *ms);
        }

    cleanup_events:
        if (measure_time) {
            SAFE_CALL(hipEventDestroy(end));
        }
    cleanup_start_event:
        if (measure_time) {
            SAFE_CALL(hipEventDestroy(start));
        }

        return err;
    }

};

#else // XPU_IS_CPU

template<typename K, typename S, typename... Args>
struct action_runner<kernel_tag, K, S, void(*)(S &, Args...)> {

    using my_type = action_runner<kernel_tag, K, S, void(*)(S &, Args...)>;

    static int call(float *ms, grid g, Args... args) {
        if (g.threads.x == -1) {
            g.threads.x = g.blocks.x;
        }
        XPU_LOG("Calling kernel '%s' [block_dim = (1, 0, 0), grid_dim = (%d, %d, %d)] with CPU driver.", type_name<K>(), g.threads.x, 0, 0);

        using clock = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<float, std::milli>;

        bool measure_time = (ms != nullptr);
        clock::time_point start;

        if (measure_time) {
            start = clock::now();
        }

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < g.threads.x; i++) {
            S smem;
            this_thread::block_idx.x = i;
            this_thread::grid_dim.x = g.threads.x;
            K::impl(smem, args...);
        }

        if (measure_time) {
            duration elapsed = clock::now() - start;
            *ms = elapsed.count();
            XPU_LOG("Kernel '%s' took %f ms", type_name<K>(), *ms);
        }

        return 0;
    }

};

#endif

template<typename A>
struct register_action {
    register_action() {
        image_context<typename A::image>::instance()->template add_symbol<A>((void *)&action_runner<A>::call);
    }
};

template<typename K, typename S>
struct register_kernel {
    register_kernel() {
        image_context<typename K::image>::instance()->template add_symbol<K>((void *)&action_runner<kernel_tag, K, S, decltype(&K::template impl<S>)>::call);
    }
};

} // namespace detail
} // namespace xpu

#if XPU_IS_CPU
#define XPU_DETAIL_TYPE_ID_MAP(image) \
    template<> \
    const char *xpu::detail::image_context<image>::file_name = XPU_IMAGE_FILE; \
    \
    template<> \
    std::unordered_map<std::string, size_t> &xpu::detail::image_context<image>::ids() { \
        static std::unordered_map<std::string, size_t> ids; \
        return ids; \
    }
#else
#define XPU_DETAIL_TYPE_ID_MAP(image)
#endif

#define XPU_DETAIL_IMAGE(image) \
    XPU_DETAIL_TYPE_ID_MAP(image); \
    template<> \
    xpu::detail::image_context<image> *xpu::detail::image_context<image>::instance() { \
        static image_context<image> ctx; \
        return &ctx; \
    } \
    extern "C" xpu::detail::image_context<image> *xpu_detail_get_context() { \
        return xpu::detail::image_context<image>::instance(); \
    }

#define XPU_DETAIL_EXPORT_FUNC(image, name, ...) \
    struct name : xpu::detail::action<image, xpu::detail::function_tag> { \
        static int impl(__VA_ARGS__); \
    }

#define XPU_DETAIL_EXPORT_CONSTANT(image, type_, name) \
    struct name : xpu::detail::action<image, xpu::detail::constant_tag> { \
        using data_t = type_; \
        static int impl(const data_t &); \
        static XPU_D const data_t &get(); \
    }

#define XPU_DETAIL_EXPORT_KERNEL(image, name, ...) \
    struct name : xpu::detail::action<image, xpu::detail::kernel_tag> { \
        template<typename S> \
        static XPU_D void impl(S &, ##__VA_ARGS__); \
    }

#if XPU_IS_CUDA

#define XPU_DETAIL_CONSTANT(name) \
    static __constant__ typename name::data_t XPU_MAGIC_NAME(xpu_detail_constant); \
    \
    int name::impl(const typename name::data_t &val) { \
        return cudaMemcpyToSymbol(XPU_MAGIC_NAME(xpu_detail_constant), &val, sizeof(name::data_t)); \
    } \
    \
    XPU_D const typename name::data_t &name::get() { \
        return XPU_MAGIC_NAME(xpu_detail_constant); \
    } \
    \
    static xpu::detail::register_action<name> XPU_MAGIC_NAME(xpu_detail_register_action){}

#elif XPU_IS_HIP

#define XPU_DETAIL_CONSTANT(name) \
    static __constant__ typename name::data_t XPU_MAGIC_NAME(xpu_detail_constant); \
    \
    int name::impl(const typename name::data_t &val) { \
        return hipMemcpyToSymbol(HIP_SYMBOL(XPU_MAGIC_NAME(xpu_detail_constant)), &val, sizeof(name::data_t)); \
    } \
    \
    XPU_D const typename name::data_t &name::get() { \
        return XPU_MAGIC_NAME(xpu_detail_constant); \
    } \
    \
    static xpu::detail::register_action<name> XPU_MAGIC_NAME(xpu_detail_register_action){}

#else // XPU_IS_CPU

#define XPU_DETAIL_CONSTANT(name) \
    static typename name::data_t XPU_MAGIC_NAME(xpu_detail_constant); \
    \
    int name::impl(const typename name::data_t &val) { \
        XPU_MAGIC_NAME(xpu_detail_constant) = val; \
        return 0; \
    } \
    \
    const typename name::data_t &name::get() { \
        return XPU_MAGIC_NAME(xpu_detail_constant); \
    } \
    \
    static xpu::detail::register_action<name> XPU_MAGIC_NAME(xpu_detail_register_action){}

#endif

#define XPU_DETAIL_FUNC(name, ...) \
    static xpu::detail::register_action<name> XPU_MAGIC_NAME(xpu_detail_register_action){}; \
    int name::impl(__VA_ARGS__)

#define XPU_DETAIL_FUNC_T(name, ...) \
    int xpu::detail::impl(__VA_ARGS__)

#define XPU_DETAIL_FUNC_TI(name) \
    static xpu::detail::register_action<name> XPU_MAGIC_NAME(xpu_detail_register_action){}

#define XPU_DETAIL_FUNC_TS(name, ...) \
    static xpu::detail::register_action<name> XPU_MAGIC_NAME(xpu_detail_register_action){}; \
    template<> int name::impl(__VA_ARGS__)

#define XPU_DETAIL_KERNEL(name, shared_memory, ...) \
    static xpu::detail::register_kernel<name, shared_memory> XPU_MAGIC_NAME(xpu_detail_register_action){}; \
    template<> XPU_D void name::impl<shared_memory>(XPU_MAYBE_UNUSED shared_memory &smem, ##__VA_ARGS__)

#define XPU_DETAIL_BLOCK_SIZE(kernel, size) \
    template<> struct xpu::block_size<kernel> : std::integral_constant<int, size> {}

#endif
