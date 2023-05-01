#ifndef XPU_DETAIL_DYNAMIC_LOADER_H
#define XPU_DETAIL_DYNAMIC_LOADER_H

#include "../defines.h"
#include "platform/cpu/this_thread.h"
#include "common.h"
#include "constant_memory.h"
#include "backend_base.h"
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

struct kernel_launch_info {
    grid g;
    void *queue_handle;
    double *ms;
};

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
    using type = int(*)(kernel_launch_info, Args...);
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


struct symbol {
    void *handle;
    std::string name;
    std::string image;
    size_t id;
};

template<typename I>
struct image_file_name {
    const char *operator()() const;
};

template <typename I>
class image_context {

public:
    static image_context<I> *instance();

    image_context() { name = type_name<I>(); }

    std::vector<symbol> &get_symbols() { return symbols; }
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
        symbols.at(id) = {
                .handle = symbol,
                .name = type_name<A>(),
                .image = name,
                .id = id
        };
    }

private:
    std::unordered_map<std::string, size_t> ids;
    std::vector<symbol> symbols;

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
        XPU_LOG("Loading '%s'", name);
        handle =
            #if defined __APPLE__
                dlopen(name, RTLD_LAZY)
            #elif defined __linux__
                dlopen(name, RTLD_LAZY | RTLD_DEEPBIND)
            #else
                #error "Unsupported platform"
            #endif
        ;

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
    typename std::enable_if_t<is_image_function_v<I, F>, int> call(Args&&... args) {
        return call_action<F>(std::forward<Args>(args)...);
    }

    template<typename C>
    typename std::enable_if_t<is_image_constant_v<I, C>, int> set(const typename C::data_t &val) {
        return call_action<C>(val);
    }

    template<typename K, typename... Args>
    typename std::enable_if_t<is_image_kernel_v<I, K>, int> run_kernel(kernel_launch_info launch_info, Args&&... args) {
        return call_action<K>(launch_info, std::forward<Args>(args)...);
    }

    void dump_symbols() {
        if (not logger::instance().active()) {
            return;
        }
        XPU_LOG("Symbols for '%s'", context->get_name().c_str());
        for (const auto &it : context->get_symbols()) {
            XPU_LOG("%zu: %s@%p [%s]", it.id, it.name.c_str(), it.handle, it.image.c_str());
        }
    }

    const std::vector<symbol> &get_symbols() const {
        return context->get_symbols();
    }

private:
    template<typename F, typename... Args>
    int call_action(Args&&... args) {
        auto *symbols = &context->get_symbols();
        size_t id = grouped_type_id<F, typename F::image>::get();
        if (id >= symbols->size()) {
            dump_symbols();
        }
        assert(id < symbols->size());
        auto symbol = symbols->at(id);
        assert(symbol.name == type_name<F>());
        auto *fn = reinterpret_cast<action_interface_t<F>>(symbol.handle);
        return fn(std::forward<Args>(args)...);
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

template<typename A, xpu::driver_t D = XPU_COMPILATION_TARGET>
struct register_action {

    static_assert(is_action_v<A>, "Type A is not a xpu::kernel, xpu::function or xpu::constant");

    using image = typename A::image;
    using tag = typename A::tag;

    register_action() {
        // printf("Registering action '%s'...\n", type_name<A>());
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
    static_assert(xpu::detail::is_device_image_v<image>, "Type passed to XPU_IMAGE is not derived from xpu::device_image..."); \
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
