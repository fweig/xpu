#ifndef XPU_DETAIL_TYPE_INFO_H
#define XPU_DETAIL_TYPE_INFO_H

#include <string>
#include <type_traits>

namespace xpu {
namespace detail {

template <typename T>
const char *type_name() noexcept {
    // TODO use std::string_view here once c++17 is available.
#ifdef __clang__
    static const char *fname = __PRETTY_FUNCTION__;
    static const size_t start = 42;
    static const size_t length = sizeof(__PRETTY_FUNCTION__) - start - 2;
#elif defined(__GNUC__)
    static const char *fname = __PRETTY_FUNCTION__;
    static const size_t start = 47;
    static const size_t length = sizeof(__PRETTY_FUNCTION__) - start - 2;
#else
    #error "Compiler not supported."
#endif

    static const std::string tname(&fname[start], length);
    return tname.c_str();
}

template<typename G>
struct type_seq {
    static size_t next() {
        static size_t value = 0;
        return value++;
    }
};

template<typename T, typename G>
struct type_id {
    static size_t get() {
        static const size_t id = type_seq<G>::next();
        return id;
    }
};

struct constant_tag {};
struct function_tag {};
struct kernel_tag {};

template<typename I, typename F>
struct is_function {
    static constexpr bool value =
        std::is_same<typename F::image, I>::value
        && std::is_same<typename F::tag, function_tag>::value;
};

template<typename M, typename F>
struct is_constant {
    static constexpr bool value = 
        std::is_same<typename F::image, M>::value
        && std::is_same<typename F::tag, constant_tag>::value;
};

template<typename I, typename F>
struct is_kernel {
    static constexpr bool value = 
        std::is_same<typename F::image, I>::value
        && std::is_same<typename F::tag, kernel_tag>::value;
};

} // namespace detail
} // namespace xpu

#endif
