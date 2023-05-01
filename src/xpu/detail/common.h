#ifndef XPU_DETAIL_COMMON_H
#define XPU_DETAIL_COMMON_H

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace xpu::detail {

struct device_image {};
template<typename T> struct is_device_image : std::is_base_of<device_image, T> {};
template<typename T> inline constexpr bool is_device_image_v = is_device_image<T>::value;

struct constant_tag {};
struct function_tag {};
struct kernel_tag {};

struct action_base {};

template<typename I, typename T>
struct action : action_base {
    static_assert(std::is_same_v<T, constant_tag> || std::is_same_v<T, function_tag> || std::is_same_v<T, kernel_tag>, "Invalid tag");
    static_assert(is_device_image_v<I>, "Invalid image");
    using image = I;
    using tag = T;
};

template<typename T> struct is_action : std::is_base_of<action_base, T> {};
template<typename T> inline constexpr bool is_action_v = is_action<T>::value;

template<typename T, typename Tag> struct has_tag : std::is_same<typename T::tag, Tag> {};
template<typename T, typename Tag> inline constexpr bool has_tag_v = has_tag<T, Tag>::value;

template<typename T> struct is_constant : std::bool_constant<is_action_v<T> && has_tag_v<T, constant_tag>> {};
template<typename T> inline constexpr bool is_constant_v = is_constant<T>::value;

template<typename I, typename T> struct is_image_constant : std::bool_constant<is_constant_v<T> && std::is_same_v<typename T::image, I>> {};
template<typename I, typename T> inline constexpr bool is_image_constant_v = is_image_constant<I, T>::value;

template<typename T> struct is_function : std::bool_constant<is_action_v<T> && has_tag_v<T, function_tag>> {};
template<typename T> inline constexpr bool is_function_v = is_function<T>::value;

template<typename I, typename T> struct is_image_function : std::bool_constant<is_function_v<T> && std::is_same_v<typename T::image, I>> {};
template<typename I, typename T> inline constexpr bool is_image_function_v = is_image_function<I, T>::value;

template<typename T> struct is_kernel : std::bool_constant<is_action_v<T> && has_tag_v<T, kernel_tag>> {};
template<typename T> inline constexpr bool is_kernel_v = is_kernel<T>::value;

template<typename I, typename T> struct is_image_kernel : std::bool_constant<is_kernel_v<T> && std::is_same_v<typename T::image, I>> {};
template<typename I, typename T> inline constexpr bool is_image_kernel_v = is_image_kernel<I, T>::value;

enum mem_type {
    mem_host,
    mem_device,
    mem_shared,
    mem_unknown,
};

enum driver_t {
    cpu,
    cuda,
    hip,
    sycl,
};
constexpr inline size_t num_drivers = 4;
const char *driver_to_str(driver_t, bool lower = false);

enum direction_t {
    dir_h2d,
    dir_d2h,
};

struct device {
    int id;
    driver_t backend;
    int device_nr;
};

struct device_prop {
    // Filled by driver
    std::string name;
    driver_t driver;
    std::string arch;
    size_t shared_mem_size;
    size_t const_mem_size;

    size_t warp_size;
    size_t max_threads_per_block;
    std::array<size_t, 3> max_grid_size;

    // Filled by runtime
    std::string xpuid;
    int id;
    int device_nr;

    size_t global_mem_total;
    size_t global_mem_available;
};

struct ptr_prop {
    mem_type type;
    device dev;
    void *ptr;
};

struct queue_handle {
    queue_handle();
    queue_handle(device dev);
    ~queue_handle();

    queue_handle(const queue_handle &) = delete;
    queue_handle &operator=(const queue_handle &) = delete;
    queue_handle(queue_handle &&) = delete;
    queue_handle &operator=(queue_handle &&) = delete;

    void *handle;
    device dev;
};

struct kernel_timings {
    std::string_view name; // Fine to make string_view, since kernel names are static
    std::vector<double> times;
    size_t bytes_input = 0;

    kernel_timings() = default;
    kernel_timings(std::string_view name) : name(name) {}
    kernel_timings(std::string_view name, double ms) : name(name) { times.emplace_back(ms); }
};

struct timings {
    std::string name;

    double wall = 0;

    bool has_details = false;
    std::vector<kernel_timings> kernels;
    double copy_h2d = 0;
    size_t bytes_h2d = 0;
    double copy_d2h = 0;
    size_t bytes_d2h = 0;
    double memset = 0;
    size_t bytes_memset = 0;

    size_t bytes_input = 0;

    std::vector<timings> children;

    timings() = default;
    timings(std::string_view name) : name(name) {}

    void merge(const timings &other);
};

using error = int;

} // namespace xpu::detail

#endif
