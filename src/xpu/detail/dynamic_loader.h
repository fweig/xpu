#ifndef XPU_DETAIL_DYNAMIC_LOADER_H
#define XPU_DETAIL_DYNAMIC_LOADER_H

#include "../common.h"
#include "../defines.h"
#include "../driver/cpu/this_thread.h"
#include "constant_memory.h"
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
#include <utility>
#include <vector>

namespace xpu::detail {

// FIXME: member_fn and action_interface belong into type_info.h
template<typename...>
struct member_fn {};

template<typename T, typename R, typename... Args>
struct member_fn<R(T::*)(Args...)> {
    using type = R(*)(Args...);
};

template<typename...>
struct action_interface {};

template<typename Tag, typename... Args>
struct action_interface<Tag, void(*)(Args...)> {
    using type = int(*)(Args...);
};

template<typename... Args>
struct action_interface<function_tag, int(*)(Args...)> {
    using type = int(*)(Args...);
};

template<typename S, typename... Args>
struct action_interface<kernel_tag, void(*)(S, Args...)> {
    using type = int(*)(float *, grid, Args...);
};

template<typename Constant>
struct action_interface<constant_tag, Constant> {
    using data_t = typename Constant::data_t;
    using type = int(*)(const data_t &);
};

template<typename T, typename A>
struct action_interface<T, A> : action_interface<T, typename member_fn<decltype(&A::operator())>::type> {};

template<typename A>
using action_interface_t = typename action_interface<typename A::tag, A>::type;

using symbol_table = std::vector<std::pair<std::string, void *>>;

template<typename I>
struct image_file_name {
    const char *operator()() const;
};

template <typename I>
class image_context {

public:
    static image_context<I> *instance();

    image_context() { name = type_name<I>(); }

    symbol_table &get_symbols() { return symbols; }
    std::string get_name() const { return name; }

    template<typename A>
    void add_symbol(void *symbol) {
        auto it = ids.find(type_name<A>());
        if (it == ids.end()) {
            ids[type_name<A>()] = grouped_type_id<A, typename A::image>::get();
        }
        size_t id = ids[type_name<A>()];
        if (symbols.size() <= id) {
            symbols.resize(id+1);
        }
        symbols.at(id) = {type_name<A>(), symbol};
    }

private:
    std::unordered_map<std::string, size_t> ids;
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
        handle = std::exchange(other.handle, nullptr);
        context = std::exchange(other.context, nullptr);
    }

    template<typename F, typename... Args>
    typename std::enable_if<is_function<I, F>::value, int>::type call(Args&&... args) {
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
        size_t id = grouped_type_id<F, typename F::image>::get();
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

template<typename F, typename... Args>
struct action_runner<function_tag, F, int(F::*)(Args...)> {
    static int call(Args... args) {
        return F{}(args...);
    }
};

#if XPU_IS_HIP_CUDA

#define SAFE_CALL(call) if (int err = call; err != 0) return err
#define ON_ERROR_GOTO(errvar, call, label) errvar = call; \
    if (errvar != 0) goto label

template<typename F, typename S, typename... Args>
__global__ void kernel_entry(Args... args) {
    using shared_memory = typename F::shared_memory;
    using context = kernel_context<shared_memory>;
    __shared__ shared_memory smem;
    F{}(context{smem}, args...);
}

template<typename F, int MaxThreadsPerBlock, typename... Args>
__global__ void __launch_bounds__(MaxThreadsPerBlock) kernel_entry_bounded(Args... args) {
    using shared_memory = typename F::shared_memory;
    using context = kernel_context<shared_memory>;
    __shared__ shared_memory smem;
    context ctx{smem};
    F{}(ctx, args...);
}

#endif

#if XPU_IS_CUDA

template<typename F>
struct action_runner<constant_tag, F> {
    using data_t = typename F::data_t;
    static int call(const data_t &val) {
        return cudaMemcpyToSymbol(constant_memory<F>, &val, sizeof(data_t));
    }
};

template<typename K, typename... Args>
struct action_runner<kernel_tag, K, void(K::*)(kernel_context<typename K::shared_memory> &, Args...)> {

    static int call(float *ms, grid g, Args... args) {
        dim block_dim = K::block_size::value;
        dim grid_dim{};

        g.get_compute_grid(block_dim, grid_dim);

        XPU_LOG("Calling kernel '%s' [block_dim = (%d, %d, %d), grid_dim = (%d, %d, %d)] with CUDA driver.", type_name<K>(), block_dim.x, block_dim.y, block_dim.z, grid_dim.x, grid_dim.y, grid_dim.z);

        bool measure_time = (ms != nullptr);
        cudaEvent_t start, end;
        int err = 0;

        if (measure_time) {
            SAFE_CALL(cudaEventCreate(&start));
            ON_ERROR_GOTO(err, cudaEventCreate(&end), cleanup_start_event);
        }

        if (measure_time) {
            ON_ERROR_GOTO(err, cudaEventRecord(start), cleanup_events);
        }

        kernel_entry_bounded<K, K::block_size::value.linear(), Args...><<<grid_dim.as_cuda_grid(), block_dim.as_cuda_grid()>>>(args...);

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

template<typename F>
struct action_runner<constant_tag, F> {
    using data_t = typename F::data_t;
    static int call(const data_t &val) {
        return hipMemcpyToSymbol(HIP_SYMBOL(constant_memory<F>), &val, sizeof(data_t));
    }
};

template<typename K, typename... Args>
struct action_runner<kernel_tag, K, void(K::*)(kernel_context<typename K::shared_memory> &, Args...)> {

    using shared_memory = typename K::shared_memory;

    static int call(float *ms, grid g, Args... args) {
        dim block_dim = K::block_size::value;
        dim grid_dim{};

        g.get_compute_grid(block_dim, grid_dim);

        XPU_LOG("Calling kernel '%s' [block_dim = (%d, %d, %d), grid_dim = (%d, %d, %d)] with HIP driver.", type_name<K>(), block_dim.x, block_dim.y, block_dim.z, grid_dim.x, grid_dim.y, grid_dim.z);

        bool measure_time = (ms != nullptr);
        hipEvent_t start, end;
        int err = 0;

        if (measure_time) {
            SAFE_CALL(hipEventCreate(&start));
            ON_ERROR_GOTO(err, hipEventCreate(&end), cleanup_start_event);
        }

        if (measure_time) {
            ON_ERROR_GOTO(err, hipEventRecord(start), cleanup_events);
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_entry_bounded<K, K::block_size::value.linear(), Args...>), grid_dim.as_cuda_grid(), block_dim.as_cuda_grid(), 0, 0, args...);
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

template<typename F>
struct action_runner<constant_tag, F> {
    using data_t = typename F::data_t;
    static int call(const data_t &val) {
        constant_memory<F> = val;
        return 0;
    }
};

template<typename K, typename... Args>
struct action_runner<kernel_tag, K, void(K::*)(kernel_context<typename K::shared_memory> &, Args...)> {

    using shared_memory = typename K::shared_memory;
    using context = kernel_context<shared_memory>;

    static int call(float *ms, grid g, Args... args) {
        dim block_dim{1, 1, 1};
        dim grid_dim{};

        g.get_compute_grid(block_dim, grid_dim);
        XPU_LOG("Calling kernel '%s' [block_dim = (%d, %d, %d), grid_dim = (%d, %d, %d)] with CPU driver.", type_name<K>(), block_dim.x, block_dim.y, block_dim.z, grid_dim.x, grid_dim.y, grid_dim.z);

        using clock = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<float, std::milli>;

        bool measure_time = (ms != nullptr);
        clock::time_point start;

        if (measure_time) {
            start = clock::now();
        }

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) collapse(3)
        #endif
        for (int i = 0; i < grid_dim.x; i++) {
            for (int j = 0; j < grid_dim.y; j++) {
                for (int k = 0; k < grid_dim.z; k++) {
                    shared_memory smem;
                    kernel_context ctx{smem};
                    this_thread::block_idx = dim{i, j, k};
                    this_thread::grid_dim = grid_dim;
                    K{}(ctx, args...);
                }
            }
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

template<typename A, xpu::driver_t D = XPU_COMPILATION_TARGET>
struct register_action {

    using image = typename A::image;
    using tag = typename A::tag;

    register_action() {
        printf("Registering action '%s'...\n", type_name<A>());
        if constexpr (std::is_same_v<tag, kernel_tag> || std::is_same_v<tag, function_tag>) {
            image_context<image>::instance()->template add_symbol<A>((void *)&action_runner<tag, A, decltype(&A::operator())>::call);
        } else if constexpr (std::is_same_v<tag, constant_tag>) {
            image_context<image>::instance()->template add_symbol<A>((void *)&action_runner<tag, A>::call);
        }
    }

    static register_action<A, D> instance;
};

template<typename A, xpu::driver_t D>
xpu::detail::register_action<A, D> xpu::detail::register_action<A, D>::instance{};

} // namespace xpu::detail

#if XPU_IS_CPU

#define XPU_DETAIL_TYPE_ID_MAP(image) \
    template<> \
    const char *xpu::detail::image_file_name<image>::operator()() const { return XPU_IMAGE_FILE; }
#define XPU_DETAIL_IMAGE_CONTEXT_GETTER(image)

#else // HIP OR CUDA

#define XPU_DETAIL_TYPE_ID_MAP(image)
#define XPU_DETAIL_IMAGE_CONTEXT_GETTER(image) \
    extern "C" xpu::detail::image_context<image> *xpu_detail_get_context() { \
        return xpu::detail::image_context<image>::instance(); \
    }

#endif

#define XPU_DETAIL_IMAGE(image) \
    XPU_DETAIL_TYPE_ID_MAP(image); \
    template<> \
    xpu::detail::image_context<image> *xpu::detail::image_context<image>::instance() { \
        static image_context<image> ctx; \
        return &ctx; \
    } \
    XPU_DETAIL_IMAGE_CONTEXT_GETTER(image) \
    void xpu_detail_dummy_func() // Force semicolon at the end of macro

#define XPU_DETAIL_EXPORT(name) \
    template struct xpu::detail::register_action<name, XPU_DETAIL_COMPILATION_TARGET>

#endif
