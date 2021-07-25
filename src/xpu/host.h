#ifndef XPU_HOST_H
#define XPU_HOST_H

#include "defines.h"
#include "common.h"
#include "detail/common.h"

#include <cstddef>
#include <cstring>
#include <iostream>
#include <utility>
#include <string>
#include <type_traits>
#include <vector>

namespace xpu {

enum direction {
    host_to_device,
    device_to_host,
};

template<class Kernel>
struct is_kernel : std::is_base_of<detail::kernel_dispatcher, Kernel> {};

class exception : public std::exception {

public:
    explicit exception(const std::string message_) : message(message_) {}

    const char *what() const noexcept override {
        return message.c_str();
    }

private:
    std::string message;

};

inline void initialize(driver);

inline void *host_malloc(size_t);
template<typename T>
T *host_malloc(size_t N) {
    return static_cast<T *>(host_malloc(sizeof(T) * N));
}

inline void *device_malloc(size_t);
template<typename T>
T *device_malloc(size_t N) {
    return static_cast<T *>(device_malloc(sizeof(T) * N));
}

inline void free(void *);
inline void memcpy(void *, const void *, size_t);
inline void memset(void *, int, size_t);

inline xpu::driver active_driver();

template<typename Kernel>
const char *get_name();

template<typename Kernel>
std::vector<float> get_timing();

template<typename Kernel, typename... Args>
void run_kernel(grid params, Args&&... args);

template<typename C>
void set_constant(const typename C::data_t &symbol);

template<typename T>
class hd_buffer {

public:
    hd_buffer() = default;
    explicit hd_buffer(size_t N);
    ~hd_buffer();

    hd_buffer<T> &operator=(const hd_buffer<T> &) = delete;
    hd_buffer<T> &operator=(hd_buffer<T> &&);

    size_t size() const { return _size; }
    T *host() { return hostdata; }
    T *device() { return devicedata; }

    bool copy_required() const { return hostdata != devicedata; }

    void reset();

private:
    size_t _size = 0;
    T *hostdata = nullptr;
    T *devicedata = nullptr;

};

template<typename T>
class d_buffer {

public:
    d_buffer() = default;
    explicit d_buffer(size_t N);
    ~d_buffer();

    d_buffer<T> &operator=(const d_buffer<T> &) = delete;
    d_buffer<T> &operator=(d_buffer<T> &&);

    size_t size() const { return _size; }
    T *data() { return devicedata; }

    void reset();

private:
    size_t _size = 0;
    T *devicedata = nullptr;

};

template<typename T>
void copy(T *dst, const T *src, size_t entries);

template<typename T>
void copy(hd_buffer<T> &buf, direction dir);

template<typename T>
void memset(hd_buffer<T> &buf, int ch);

template<typename T>
void memset(d_buffer<T> &buf, int ch);

} // namespace xpu

#include "host_impl.h"

#endif
