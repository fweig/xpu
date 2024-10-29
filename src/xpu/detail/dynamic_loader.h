#ifndef XPU_DETAIL_DYNAMIC_LOADER_H
#define XPU_DETAIL_DYNAMIC_LOADER_H

#include "../defines.h"
#include "../common.h"
#include "common.h"
#include "log.h"
#include "type_info.h"

#if XPU_IS_HIP
#include <hip/hip_runtime_api.h>
#endif

#include <dlfcn.h>

#include <algorithm>
#include <cassert>
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

class symbol_table {

public:
    template<typename I, xpu::driver_t D>
    static symbol_table &instance() {
        static symbol_table _instance{type_name<I>(), D};
        return _instance;
    }

    symbol_table(const char *name, xpu::driver_t driver) : m_name(name), m_driver(driver) {}

    std::string name() const { return m_name; }
    xpu::driver_t driver() const { return m_driver; }

    template<typename A>
    void add(void *symbol) {

        auto it = m_symbols.find(type_name<A>());
        if (it != m_symbols.end()) {
            XPU_LOG("Symbol '%s' already exists in '%s'", type_name<A>(), m_name.c_str());
            return;
        }

        size_t id = grouped_type_id<A, typename A::image>::get();
        const char *name = type_name<A>();

        m_symbols[name] = {
                .handle = symbol,
                .name = name,
                .image = m_name,
                .id = id
        };
    }

    std::vector<symbol> linearize() const {
        std::vector<symbol> result;
        result.reserve(m_symbols.size());
        for (const auto &pair : m_symbols) {
            result.push_back(pair.second);
        }
        std::sort(result.begin(), result.end(), [](const symbol &a, const symbol &b) {
            return a.id < b.id;
        });
        return result;
    }

    std::vector<symbol> linearize_with(const std::vector<symbol> &gt) const {
        std::vector<symbol> result;
        for (size_t i = 0; i < gt.size(); ++i) {
            assert(gt[i].id == i);
            assert(gt[i].image == m_name);
            auto it = m_symbols.find(gt[i].name);
            assert(it != m_symbols.end());
            result.push_back(it->second);
            result.back().id = i;
        }
        return result;
    }

private:
    std::unordered_map<std::string, symbol> m_symbols;
    std::string m_name;
    xpu::driver_t m_driver;

};

template<typename I>
class image {

private:
    void *m_handle = nullptr;
    symbol_table *m_context = nullptr;
    std::vector<symbol> m_symbols;

public:
    image() {
        m_context = &symbol_table::instance<I, xpu::cpu>();
        m_symbols = m_context->linearize();
    }

    image(const char *name) {
        XPU_LOG("Loading '%s'", name);
        m_handle =
            #if defined __APPLE__
                dlopen(name, RTLD_LAZY)
            #elif defined __linux__
                dlopen(name, RTLD_LAZY | RTLD_DEEPBIND)
            #else
                #error "Unsupported platform"
            #endif
        ;

        if (m_handle == nullptr) {
            XPU_LOG("Error opening '%s: %s", name, dlerror());
        }
        assert(m_handle != nullptr);
        auto *get_context = reinterpret_cast<symbol_table *(*)()>(dlsym(m_handle, "xpu_detail_get_context"));
        assert(get_context != nullptr);
        m_context = get_context();
        assert(m_context->name() == type_name<I>());

        auto &cpu_context = symbol_table::instance<I, xpu::cpu>();
        auto cpu_symbols = cpu_context.linearize();
        m_symbols = m_context->linearize_with(cpu_symbols);
    }

    ~image() {
        if (m_handle != nullptr) {
            dlclose(m_handle);
        }
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
        XPU_LOG("Symbols for '%s'", m_context->name().c_str());
        for (const auto &it : m_symbols) {
            XPU_LOG("%zu: %s@%p [%s]", it.id, it.name.c_str(), it.handle, it.image.c_str());
        }
    }

    const std::vector<symbol> &symbols() const {
        return m_symbols;
    }

private:
    template<typename F, typename... Args>
    int call_action(Args&&... args) {
        size_t id = grouped_type_id<F, typename F::image>::get();
        if (id >= m_symbols.size()) {
            dump_symbols();
        }
        assert(id < m_symbols.size());
        const auto &symbol = m_symbols.at(id);
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
            symbol_table::instance<image, D>().template add<A>((void *)&action_runner<tag, A, decltype(&A::operator())>::call);
        } else if constexpr (std::is_same_v<tag, constant_tag>) {
            symbol_table::instance<image, D>().template add<A>((void *)&action_runner<tag, A>::call);
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
#define XPU_DETAIL_symbol_table_GETTER(image)

#else // HIP OR CUDA

#define XPU_DETAIL_TYPE_ID_MAP(image)
#define XPU_DETAIL_symbol_table_GETTER(image) \
    extern "C" xpu::detail::symbol_table *xpu_detail_get_context() { \
        return &xpu::detail::symbol_table::instance<image, XPU_COMPILATION_TARGET>(); \
    }

#endif // XPU_IS_CPU

#define XPU_DETAIL_IMAGE(image) \
    static_assert(xpu::detail::is_device_image_v<image>, "Type passed to XPU_IMAGE is not derived from xpu::device_image..."); \
    XPU_DETAIL_TYPE_ID_MAP(image); \
    XPU_DETAIL_symbol_table_GETTER(image) \
    void xpu_detail_dummy_func() // Force semicolon at the end of macro

#define XPU_DETAIL_EXPORT(name) \
    template struct xpu::detail::register_action<name, XPU_DETAIL_COMPILATION_TARGET>

#endif
