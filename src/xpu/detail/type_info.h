#ifndef XPU_DETAIL_TYPE_INFO_H
#define XPU_DETAIL_TYPE_INFO_H

#include <string>
#include <type_traits>

namespace xpu::detail {

template <typename T>
const char *type_name() noexcept {
    // Not using std::string_view here, because it creates a lot of problems with the XPU_LOG macro
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

template<typename Group = void>
struct type_seq {
    static size_t next() {
        static size_t value = 0;
        return value++;
    }
};

// xpu needs to two different type ids internally.
// A global one, and one that splits types into groups instead.
// The latter is used to assign kernels ids that are unique only in their specific
// and to exclude image ids from global ids. While the former allows mapping all kernels
// onto a global unique id.

template<typename T, typename Group>
struct grouped_type_id {
    static_assert(not std::is_same_v<Group, void>);

    static size_t get() {
        static const size_t id = type_seq<Group>::next();
        return id;
    }
};

template<typename T>
struct linear_type_id {
    static size_t get() {
        static const size_t id = type_seq<>::next();
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

struct internal_ctor_t {};
constexpr inline internal_ctor_t internal_ctor{};

} // namespace xpu::detail

#endif
