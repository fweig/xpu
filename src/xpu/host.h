#ifndef XPU_HOST_H
#define XPU_HOST_H

#include "defines.h"
#include "common.h"
#include "detail/common.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <utility>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace xpu {

// Forward declarations
template<typename T> class buffer_prop;
class queue;
class ptr_prop;
namespace detail { class runtime; }

enum direction {
    host_to_device,
    device_to_host,
};

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
        // Use c functions for output to avoid including iostream in host.h ...
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
 * @brief Preload the given device image.
 * @tparam I Device image type.
 * This call is optional. If not preloaded, the device image will be loaded
 * automatically when the first kernel is launched.
 */
template<typename I>
void preload();

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

/**
 * @brief Allocate the stack memory on the device.
 */
void stack_alloc(size_t size);

/**
 * @brief Pop entries from the stack.
 * @param head Pointer to the stack entry to pop or nullptr to pop the entire stack.
 */
void stack_pop(void *head=nullptr);

/**
 * Represents a device found on the system.
 */
class device {

public:
    /**
     * @brief Get all available devices.
     */
    static std::vector<device> all();

    /**
     * @brief Get the active device.
     */
    static device active();

    /**
     * @brief Construct CPU device.
     */
    device();

    /**
     * @brief Lookup device by string.
     * @see xpu::settings::device for possible values.
     */
    explicit device(std::string_view xpuid);

    /**
     * @brief Construct device from device id.
     * @param id Device id.
     * The device id is a unique number assigned to each device.
     * It is also the index of the device in the vector returned by device::all().
     */
    explicit device(int id);

    /**
     * @brief Construct device from driver and device number.
     */
    explicit device(driver_t driver, int device_nr);

    device(const device &) = default;
    device(device &&) = default;
    device &operator=(const device &) = default;
    device &operator=(device &&) = default;

    /**
     * @brief Get the device id.
     */
    int id() const { return m_impl.id; }

    /**
     * @brief Get the backend associated with the device.
     */
    driver_t backend() const { return static_cast<driver_t>(m_impl.backend); }

    /**
     * @brief Get the device number within the backend.
     */
    int device_nr() const { return m_impl.device_nr; }

private:
    friend class queue;
    friend class ptr_prop;
    explicit device(detail::device impl) : m_impl(std::move(impl)) {}
    detail::device m_impl;

};

/**
 * Device properties.
 */
class device_prop {

public:
    device_prop() = delete;

    /**
     * @brief Query properties of the given device.
     */
    device_prop(device);

    /**
     * @brief Get the name of the device.
     */
    std::string_view name() const { return m_prop.name; }

    /**
     * @brief Get the backend associated with the device.
     */
    driver_t backend() const { return static_cast<driver_t>(m_prop.driver); }

    /**
     * @brief Returns the architecture of the device, if applicable.
     */
    std::string_view arch() const { return m_prop.arch; }

    /**
     * @brief Returns the size of shared memory per block in bytes.
     */
    size_t shared_mem_size() const { return m_prop.shared_mem_size; }

    /**
     * @brief Returns the size of constant memory in bytes.
     */
    size_t const_mem_size() const { return m_prop.const_mem_size; }

    /**
     * @brief Returns the number of threads in a warp.
     */
    size_t warp_size() const { return m_prop.warp_size; }

    /**
     * @brief Returns the max number of threads in a block.
     */
    size_t max_threads_per_block() const { return m_prop.max_threads_per_block; }

    /**
     * @brief Returns the max number of threads in a block.
     */
    std::array<size_t, 3> max_grid_size() const { return m_prop.max_grid_size; }

    /**
     * @brief Get the string used to identify the device.
     * @see xpu::settings::device for possible values.
     */
    std::string_view xpuid() const { return m_prop.xpuid; }

    /**
     * @brief Get the device id.
     */
    int id() const { return m_prop.id; }

    /**
     * @brief Get the device number within the backend.
     */
    int device_nr() const { return m_prop.device_nr; }

    /**
     * @brief Returns the total amount of global memory in bytes.
     */
    size_t global_mem_total() const { return m_prop.global_mem_total; }

    /**
     * @brief Returns the amount of global memory available in bytes.
     */
    size_t global_mem_available() const { return m_prop.global_mem_available; }

private:
    detail::device_prop m_prop;
};

class queue {

public:
    /**
     * Construct a queue for the standard device set via the XPU_DEVICE environment variable.
     */
    queue();

    /**
     * Construct a queue for the given device.
     */
    explicit queue(device);

    void copy(const void *from, void *to, size_t size);

    template<typename T>
    void copy(buffer<T>, direction);

    void memset(void *dst, int value, size_t size);

    template<typename T>
    void memset(buffer<T>, int value);

    template<typename Kernel, typename... Args>
    void launch(grid params, Args&&... args);

    void wait();

private:
    std::shared_ptr<detail::queue_handle> m_handle;

};

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

          T *begin() { return m_data; }
    const T *begin() const { return m_data; }
          T *end() { return m_data + m_size; }
    const T *end() const { return m_data + m_size; }

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
    host = detail::mem_host,

    /**
     * Memory allocated on the device by the GPU driver.
     */
    device = detail::mem_device,

    /**
     * Memory allocated on the host and device by the GPU driver.
     * GPU driver will synchronise data to the device when needed.
     * (also known as unified or managed memory)
     */
    shared = detail::mem_shared,

    /**
     * Uknown memory type.
     * Usually memory allocated on the host by libc.
     * Can't be accessed on the device.
     */
    unknown = detail::mem_unknown,
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
    void *ptr() const { return m_prop.ptr; }

    /**
     * @returns The type of the memory.
     * @see mem_type
     */
    mem_type type() const { return static_cast<mem_type>(m_prop.type); }

    /**
     * @returns The device the memory is allocated on.
     */
    xpu::device device() const { return xpu::device{m_prop.dev}; }

    /**
     * @returns The backend used to allocate the memory.
     * @see driver_t
     */
    driver_t backend() const { return static_cast<driver_t>(m_prop.dev.backend); }

private:
    detail::ptr_prop m_prop;
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

template<typename T>
void copy(T *dst, const T *src, size_t entries);

template<typename T>
void copy(buffer<T> &buf, direction dir);

} // namespace xpu

#include "host.tpp"

#endif
