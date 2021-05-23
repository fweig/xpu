#ifndef XPU_DETAIL_DYNAMIC_LOADER_H
#define XPU_DETAIL_DYNAMIC_LOADER_H

#include "../common.h"
#include "../defines.h"
#include "macros.h"
#include "../driver/cpu/this_thread.h"
#include "type_info.h"

#include <dlfcn.h>

#include <cassert>
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
    using type = void(*)(Args...);
};

template<typename S, typename... Args>
struct action_interface<kernel_tag, void(*)(S, Args...)> {
    using type = void(*)(grid, Args...);
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
        handle = dlopen(name, RTLD_LAZY | RTLD_DEEPBIND);
        if (handle == nullptr) {
            std::cout << "Error openening " << name << ": " << dlerror() << std::endl; 
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
        call_action<F>(args...);
    }

    template<typename F>
    typename std::enable_if<is_constant<I, F>::value>::type set(const typename F::data_t &val) {
        call_action<F>(val);
    }

    template<typename K, typename... Args>
    typename std::enable_if<is_kernel<I, K>::value>::type run_kernel(grid g, Args&&... args) {
        call_action<K>(g, args...);
    }

    void dump_symbols() {
        for (auto it : context->get_symbols()) {
            std::cout << it.first << ": " << it.second << std::endl;
        }
    }

private:
    template<typename F, typename... Args>
    void call_action(Args... args) {
        std::cout << "Calling kernel " << type_name<F>() << std::endl;
        auto *symbols = &context->get_symbols();
        size_t id = type_id<F, typename F::image>::get();
        if (id >= symbols->size()) {
            dump_symbols();
        }
        assert(id < symbols->size());
        auto symbol = symbols->at(id);
        assert(symbol.first == type_name<F>());
        auto *fn = reinterpret_cast<action_interface_t<F>>(symbol.second);
        fn(args...);
    }

};

template<typename...>
struct action_runner {};

template<typename Tag, typename F, typename... Args>
struct action_runner<Tag, F, void(*)(Args...)> {

    using my_type = action_runner<Tag, F, void(*)(Args...)>;

    static void call(Args... args) {
        printf("My type is: %s\n", type_name<my_type>());
        F::impl(args...);
    }
};

template<typename F>
struct action_runner<F> : action_runner<typename F::tag, F, decltype(&F::impl)> {};

#if XPU_IS_CUDA

template<typename F, typename S, typename... Args>
__global__ void kernel_entry(Args... args) {
    __shared__ S smem;
    F::impl(smem, args...);
}

template<typename K, typename S, typename... Args>
struct action_runner<kernel_tag, K, S, void(*)(S &, Args...)> {

    using my_type = action_runner<kernel_tag, K, S, void(*)(S &, Args...)>;

    static void call(grid g, Args... args) {
        if (g.blocks.x == -1) {
            g.blocks.x = (g.threads.x + 63) / 64;
        }
        kernel_entry<K, S, Args...><<<g.blocks.x, 64>>>(args...);
        cudaDeviceSynchronize();
    }

};

#elif XPU_IS_HIP

#error "Hip kernel runner not implemented!"

#else // XPU_IS_CPU

template<typename K, typename S, typename... Args>
struct action_runner<kernel_tag, K, S, void(*)(S &, Args...)> {

    using my_type = action_runner<kernel_tag, K, S, void(*)(S &, Args...)>;

    static void call(grid g, Args... args) {
        if (g.threads.x == -1) {
            g.threads.x = g.blocks.x;
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

#define XPU_IMAGE(image) \
    XPU_DETAIL_TYPE_ID_MAP(image); \
    template<> \
    xpu::detail::image_context<image> *xpu::detail::image_context<image>::instance() { \
        static image_context<image> ctx; \
        return &ctx; \
    } \
    extern "C" xpu::detail::image_context<image> *xpu_detail_get_context() { \
        return xpu::detail::image_context<image>::instance(); \
    }

#define XPU_EXPORT_FUNC(image, name, ...) \
    struct name : xpu::detail::action<image, xpu::detail::function_tag> { \
        static void impl(__VA_ARGS__); \
    }

#define XPU_EXPORT_CONSTANT(image, type_, name) \
    struct name : xpu::detail::action<image, xpu::detail::constant_tag> { \
        using data_t = type_; \
        static void impl(const data_t &); \
        static XPU_D const data_t &get(); \
    }

#define XPU_EXPORT_KERNEL(image, name, ...) \
    struct name : xpu::detail::action<image, xpu::detail::kernel_tag> { \
        template<typename S> \
        static XPU_D void impl(S &, ##__VA_ARGS__); \
    }

#if XPU_IS_CUDA

#define XPU_CONSTANT(name) \
    __constant__ typename name::data_t XPU_MAGIC_NAME(xpu_detail_constant); \
    \
    void name::impl(const typename name::data_t &val) { \
        cudaMemcpyToSymbol(XPU_MAGIC_NAME(xpu_detail_constant), &val, sizeof(name::data_t)); \
    } \
    \
    XPU_D const typename name::data_t &name::get() { \
        return XPU_MAGIC_NAME(xpu_detail_constant); \
    } \
    \
    static xpu::detail::register_action<name> XPU_MAGIC_NAME(xpu_detail_register_action){}

#elif XPU_IS_HIP

#error "XPU_CONSTANT is not implemented for HIP."

#else // XPU_IS_CPU

#define XPU_CONSTANT(name) \
    static typename name::data_t XPU_MAGIC_NAME(xpu_detail_constant); \
    \
    void name::impl(const typename name::data_t &val) { \
        XPU_MAGIC_NAME(xpu_detail_constant) = val; \
    } \
    \
    const typename name::data_t &name::get() { \
        return XPU_MAGIC_NAME(xpu_detail_constant); \
    } \
    \
    static xpu::detail::register_action<name> XPU_MAGIC_NAME(xpu_detail_register_action){}

#endif

#define XPU_FUNC(name, ...) \
    static xpu::detail::register_action<name> XPU_MAGIC_NAME(xpu_detail_register_action){}; \
    void name::impl(__VA_ARGS__)

#define XPU_FUNC_T(name, ...) \
    void xpu::detail::impl(__VA_ARGS__)

#define XPU_FUNC_TI(name) \
    static xpu::detail::register_action<name> XPU_MAGIC_NAME(xpu_detail_register_action){}

#define XPU_FUNC_TS(name, ...) \
    static xpu::detail::register_action<name> XPU_MAGIC_NAME(xpu_detail_register_action){}; \
    template<> void name::impl(__VA_ARGS__)

#define XPU_KERNEL(name, shared_memory, ...) \
    static xpu::detail::register_kernel<name, shared_memory> XPU_MAGIC_NAME(xpu_detail_register_action){}; \
    template<> XPU_D void name::impl<shared_memory>(XPU_MAYBE_UNUSED shared_memory &smem, ##__VA_ARGS__)

#endif
