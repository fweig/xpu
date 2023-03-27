#ifndef XPU_HOST_H
#define XPU_HOST_H

#include "defines.h"
#include "common.h"
#include "detail/common.h"
#include "detail/mem_type.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <utility>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace xpu {

// Forward declarations
template<typename T> class buffer_prop;
namespace detail { class runtime; }

enum direction {
    host_to_device,
    device_to_host,
};

template<class Kernel>
struct is_kernel : std::is_base_of<detail::kernel_dispatcher, Kernel> {};

class exception : public std::exception {

public:
    explicit exception(std::string_view message_) : message(message_) {}

    const char *what() const noexcept override { return message.c_str(); }

private:
    std::string message;

};


/**
 * @brief Settings used to initialize xpu.
 */
struct settings {
    /**
     * @brief Select the default device to use.
     * Values must have the form "`<driver><devicenr>`".
     * If `devicenr` is missing, defaults to device 0 of selected driver.
     * Possible values are for example: `cpu`, `cuda0`, `cuda1`, `hip0`, `sycl1`.
     * Value may be overwritten by setting environment variable XPU_DEVICE.
     */
    std::string device = "cpu";

    /**
     * @brief Enable internal logging.
     * Display information about device operations like memory allocation,
     * kernel launches, memory transfers. This is useful for debugging.
     * Value may be overwritten by setting environment variable XPU_VERBOSE.
     * See also 'logging_sink' field for customizing the output.
     */
    bool verbose = false;

    /**
     * @brief Set a custom logging sink.
     * By default messages are written to stderr. Has no effect if 'verbose' is false.
     */
    std::function<void(std::string_view)> logging_sink = [](std::string_view msg) {
        // Use c functions for output to avoid including iostream...
        std::fwrite(msg.data(), 1, msg.size(), stderr);
        std::fputc('\n', stderr);
    };

    /**
     * @brief Enable profiling of kernels.
     * Use get_timing<KernelName>() to retrieve the timing information.
     * Value may be overwritten by setting environment variable XPU_PROFILE.
     */
    bool profile = false;
};

/**
 * @brief Initialize xpu.
 * @param settings Settings to use.
 * Initializes xpu runtime with the given settings.
 * Should be called once at the beginning of the program.
 * Before any other xpu functions are called.
 *
 * @see xpu::settings
 */
inline void initialize(settings = {});

/**
 * @brief Allocate memory on the device.
 * @param size Size of the memory to allocate in bytes.
 */
void *malloc_device(size_t size);

/**
 * @brief Allocate memory on the device.
 * @param elems Number of elements to allocate.
 * @tparam T Type of the memory to allocate.
 * @note The memory is not initialized.
 */
template<typename T>
T *malloc_device(size_t elems);

/**
 * @brief Allocate pinned memory on the host that can be accessed by the device.
 * @param size Size of the memory to allocate in bytes.
 */
void *malloc_host(size_t size);

/**
 * @brief Allocate pinned memory on the host that can be accessed by the device.
 * @param elems Number of elements to allocate.
 * @tparam T Type of the memory to allocate.
 * @note The memory is not initialized.
 */
template<typename T>
T *malloc_host(size_t elems);

/**
 * @brief Allocate memory that can be accessed by the device and the host.
 * @param size Size of the memory to allocate in bytes.
 */
void *malloc_shared(size_t);

/**
 * @brief Allocate memory that can be accessed by the device and the host.
 * @param elems Number of elements to allocate.
 * @tparam T Type of the memory to allocate.
 * @note The memory is not initialized.
 */
template<typename T>
T *malloc_shared(size_t);

/**
 * @brief Free memory allocated with malloc_device, malloc_host or malloc_shared.
 * @param ptr Pointer to the memory to free.
 */
inline void free(void *);

inline void memcpy(void *, const void *, size_t);
inline void memset(void *, int, size_t);

inline std::vector<device_prop> get_devices();
inline device_prop device_properties();
inline device_prop device_properties(driver_t backend, int device);
inline xpu::driver_t active_driver();

template<typename Kernel>
const char *get_name();

template<typename Kernel>
std::vector<float> get_timing();

template<typename Kernel, typename... Args>
void run_kernel(grid params, Args&&... args);

template<typename Func, typename... Args>
void call(Args&&... args);

template<typename C>
void set_constant(const typename C::data_t &symbol);

/**
 * @brief Create a view from a buffer.
 * Create a view from a buffer to access the underlying data on the host.
 * The view is a lightweight wrapper around the buffer and does not own the data.
 * If the underlying buffer can't be accessed on the host, an runtime_error is thrown.
 * Note that no synchronization with the device is performed, so the data may be out of date.
 */
template<typename T>
class h_view {

public:
    /**
     * @brief Create an empty view.
     */
    h_view() = default;

    /**
     * @brief Create a view from a buffer.
     */
    explicit h_view(buffer<T> &);

    /**
     * @brief Create a view from buffer properties.
     */
    explicit h_view(const buffer_prop<T> &);

    /**
     * @returns Pointer to the underlying data.
     */
    T *data() const { return m_data; }

    /**
     * @returns Size of the view in number of elements.
     */
    size_t size() const { return m_size; }

    /**
     * Check if the view is empty.
     */
    bool empty() const { return m_size == 0; }

    /**
     * @returns Reference to the element at index i.
     * @note This call is always bounds checked. Use instead unsafe_at() if no bounds check are needed.
     * Equivalent to calling at(i).
     */
          T &operator[](size_t i);
    const T &operator[](size_t i) const;

    /**
     * @returns Reference to the element at index i.
     * @note This call is always bounds checked. Use instead unsafe_at() if no bounds check are needed.
     * Equivalent to []-operator.
     */
          T &at(size_t i);
    const T &at(size_t i) const;

    /**
     * @returns Reference to the element at index i.
     * @note No bounds checking is performed. Usually you want to use at() instead.
     */
          T &unsafe_at(size_t i) { return m_data[i]; }
    const T &unsafe_at(size_t i) const { return m_data[i]; }

private:
    T *m_data = nullptr;
    size_t m_size = 0;
};

/**
 * Different types of allocated memory.
 */
enum class mem_type {
    /**
     * Memory allocated on the host by the GPU driver.
     * Can be accessed by the device. (also known as pinned memory)
     */
    host = detail::host,

    /**
     * Memory allocated on the device by the GPU driver.
     */
    device = detail::device,

    /**
     * Memory allocated on the host and device by the GPU driver.
     * GPU driver will synchronise data to the device when needed.
     * (also known as unified or managed memory)
     */
    shared = detail::shared,

    /**
     * Uknown memory type.
     * Usually memory allocated on the host by libc.
     * Can't be accessed on the device.
     */
    unknown = detail::unknown,
};

/**
 * @brief Properties of a pointer.
 * Properties of a pointer allocated with malloc_device, malloc_host or malloc_shared.
 */
class ptr_prop {

public:
    ptr_prop() = delete;

    /**
     * @brief Create a pointer property object from a pointer.
     * @param ptr Pointer to the memory.
     */
    explicit ptr_prop(const void *);

    /**
     * @returns The pointer.
     */
    void *ptr() const { return m_ptr; }

    /**
     * @returns The type of the memory.
     * @see mem_type
     */
    mem_type type() const { return m_type; }

    /**
     * @returns The device the memory is allocated on.
     */
    int device() const { return m_device; }

    /**
     * @returns The backend used to allocate the memory.
     * @see driver_t
     */
    driver_t backend() const { return m_backend; }

private:
    friend class detail::runtime;
    void *m_ptr;
    int m_device;
    xpu::driver_t m_backend;
    mem_type m_type;
};

template<typename T>
class buffer_prop {

public:
    buffer_prop() = delete;
    explicit buffer_prop(const buffer<T> &);

    size_t size() const { return m_size_bytes / sizeof(T); }
    size_t size_bytes() const { return m_size_bytes; }
    buffer_type type() const { return m_type; }
    T *h_ptr() const { return m_host; }
    T *d_ptr() const { return m_device; }

    h_view<T> view() const { return h_view<T>{*this}; }

private:
    size_t m_size_bytes;
    T *m_host;
    T *m_device;
    buffer_type m_type;
};

// DEPRECATED - use buffer instead
template<typename T>
class hd_buffer {

public:
    hd_buffer() = default;
    explicit hd_buffer(size_t N);
    ~hd_buffer();

    hd_buffer<T> &operator=(const hd_buffer<T> &) = delete;
    hd_buffer<T> &operator=(hd_buffer<T> &&);

    size_t size() const { return m_size; }

          T *h()       { return m_h; }
    const T *h() const { return m_h; }

          T *d()       { return m_d; }
    const T *d() const { return m_d; }

          T &operator[](size_t idx)       { return m_h[idx]; }
    const T &operator[](size_t idx) const { return m_h[idx]; }

    bool copy_required() const { return m_h != m_d; }

    void reset();

private:
    size_t m_size = 0;
    T *m_h = nullptr;
    T *m_d = nullptr;

};

// DEPRECATED - use buffer instead
template<typename T>
class d_buffer {

public:
    d_buffer() = default;
    explicit d_buffer(size_t N);
    ~d_buffer();

    d_buffer<T> &operator=(const d_buffer<T> &) = delete;
    d_buffer<T> &operator=(d_buffer<T> &&);

    size_t size() const { return m_size; }

          T *d()       { return m_d; }
    const T *d() const { return m_d; }

    void reset();

private:
    size_t m_size = 0;
    T *m_d = nullptr;

};

template<typename T>
void copy(T *dst, const T *src, size_t entries);

template<typename T>
void copy(hd_buffer<T> &buf, direction dir);

template<typename T>
void copy(buffer<T> &buf, direction dir);

template<typename T>
void memset(hd_buffer<T> &buf, int ch);

template<typename T>
void memset(d_buffer<T> &buf, int ch);

} // namespace xpu

#include "host_impl.h"

#endif
