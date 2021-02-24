#ifndef XPU_HOST_H
#define XPU_HOST_H

#include "defs.h"

#include <cstddef>
#include <cstring>
#include <iostream>
#include <utility>
#include <string>
#include <type_traits>

namespace xpu {

enum class driver {
    cpu,
    cuda,
    hip,
};

enum class side {
    host,
    device,
};

enum direction {
    host_to_device,
    device_to_host,
};

struct lane {
    struct standard_t {};
    static constexpr standard_t standard{};

    struct cpu_t {};
    static constexpr cpu_t cpu{};

    int value;

    lane(standard_t) : value(0) {}
    lane(cpu_t) : value(0) {}
};

struct dim {
    int x = 0; 
    int y = 0; 
    int z = 0;

    XPU_D dim(int _x) : x(_x) {}
    XPU_D dim(int _x, int _y) : x(_x), y(_y) {}
    XPU_D dim(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}
};

struct grid {

    static inline grid n_blocks(dim blocks, lane l = lane::standard) {
        return grid{blocks, dim{-1}, l};
    }

    static inline grid n_threads(dim threads, lane l = lane::standard) {
        return grid{dim{-1}, threads, l};
    }

    static inline grid fill(lane l = lane::standard);

    dim blocks;
    dim threads;

private:
    grid(dim b, dim t, lane) : blocks(b), threads(t) {} 

};

// internal definitions.
using error = int;

class driver_interface {

public:
    virtual xpu::error setup() = 0;
    virtual xpu::error device_malloc(void **, size_t) = 0;
    virtual xpu::error free(void *) = 0;
    virtual xpu::error memcpy(void *, const void *, size_t) = 0;
    virtual xpu::error memset(void *, int, size_t) = 0;

};

struct kernel_info {
    dim i_thread;
    dim n_threads;
    dim i_block;
    dim n_blocks;
};

struct kernel_dispatcher_any {};

template<typename K, typename L>
struct kernel_dispatcher : kernel_dispatcher_any {
    using library = L;
    using kernel = K;

    template<typename... Args>
    static inline void dispatch(library &inst, grid params, Args &&... args) {
        kernel::dispatch_impl(inst, params, std::forward<Args>(args)...);
    }

    static inline const char *name() {
        return kernel::name_impl();
    }
};

template<class K>
struct is_kernel : std::is_base_of<kernel_dispatcher_any, K> {};

// Some utility classes for loading shared libraries at runtime
class library_loader {

public:
    library_loader(const std::string &);
    ~library_loader();

    void *symbol(const std::string &);

private:
    void *handle = nullptr;

};

template<class T>
class lib_obj {

public:
    using create_f = T*();
    using destroy_f = void(T*);

    T *obj = nullptr;

    lib_obj(const std::string &libname) : lib(libname) {
        create = reinterpret_cast<create_f *>(lib.symbol("create"));
        destroy = reinterpret_cast<destroy_f *>(lib.symbol("destroy"));
        obj = create();
    }

    ~lib_obj() {
        if (obj != nullptr) {
            destroy(obj);
        }
    }

private:
    library_loader lib;
    create_f *create = nullptr;
    destroy_f *destroy = nullptr;

};

// library interface
class exception : public std::exception {

public:
    explicit exception(const std::string message_) : message(message_) {}

    const char *what() const noexcept override {
        return message.c_str();
    }

private:
    std::string message;

};

void initialize(driver);

void *host_malloc(size_t);
template<typename T>
T *host_malloc(size_t N) {
    return static_cast<T *>(host_malloc(sizeof(T) * N));
}

void *device_malloc(size_t);
template<typename T>
T *device_malloc(size_t N) {
    return static_cast<T *>(device_malloc(sizeof(T) * N));
}

template<typename T>
T *malloc(size_t N, side where) {
    switch (where) {
        case side::device:
            return device_malloc<T>(N);
        case side::host:
            return host_malloc<T>(N);
    }
}

void free(void *);
void memcpy(void *, const void *, size_t);
void memset(void *, int, size_t);

driver active_driver();

template<typename Kernel, typename Enable = typename std::enable_if<is_kernel<Kernel>::value>::type>
const char *get_name() {
    return Kernel::name();
}

template<typename Kernel, typename Enable = typename std::enable_if<is_kernel<Kernel>::value>::type, typename... Args>
void run_kernel(grid params, Args&&... args) {
    std::string backend = "CPU";
    if (active_driver() == driver::cuda) {
        backend = "CUDA";
    }
    std::cout << "Running kernel " << get_name<Kernel>() << " on backend " << backend << std::endl;
    Kernel::dispatch(Kernel::library::instance(active_driver()), params, std::forward<Args>(args)...);
}

template<typename DeviceLibrary, typename C>
void set_cmem(const C &symbol) {
    DeviceLibrary::template cmem<C>::set(DeviceLibrary::instance(active_driver()), symbol);
}

template<typename T>
class hd_buffer {

public:
    hd_buffer() = default;
    explicit hd_buffer(size_t N) {
        std::cout << "Allocate hd_buffer with " << N << " elems of size " << sizeof(T) << std::endl; 
        _size = N;
        hostdata = static_cast<T *>(std::malloc(sizeof(T) * N));
        std::cout << "Finished allocation" << std::endl;

        if (active_driver() == xpu::driver::cpu) {
            devicedata = hostdata;
        } else {
            devicedata = device_malloc<T>(N);
        }
    }

    ~hd_buffer() {
        if (hostdata != nullptr) {
            std::free(hostdata);
        }
        if (copy_required()) {
            xpu::free(devicedata);
        }
    }

    hd_buffer<T> &operator=(const hd_buffer<T> &) = delete;
    hd_buffer<T> &operator=(hd_buffer<T> &&other) {
        _size = other._size;
        hostdata = other.hostdata;
        devicedata = other.devicedata;

        other._size = 0;
        other.hostdata = other.devicedata = nullptr;
    }

    size_t size() const { return _size; }
    T *host() { return hostdata; }
    T *device() { return devicedata; }

    bool copy_required() const { return hostdata != devicedata; }

private:
    size_t _size = 0;
    T *hostdata = nullptr;
    T *devicedata = nullptr;

};

template<typename T>
class d_buffer {

public:
    d_buffer() = default;
    explicit d_buffer(size_t N) {
        _size = N;
        devicedata = device_malloc<T>(N);
    }

    ~d_buffer() {
        free(devicedata);
    }

    d_buffer<T> &operator=(const d_buffer<T> &) = delete;
    d_buffer<T> &operator=(d_buffer<T> &&other) {
        _size = other._size;
        devicedata = other.devicedata;

        other._size = 0;
        other.devicedata = nullptr;
    }

    size_t size() const { return _size; }
    T *data() { return devicedata; }

private:
    size_t _size = 0;
    T *devicedata = nullptr;

};

template<typename T>
void copy(T *dst, const T *src, size_t entries) {
    xpu::memcpy(dst, src, sizeof(T) * entries);
}

template<typename T>
void copy(hd_buffer<T> &buf, direction dir) {
    if (not buf.copy_required()) {
        return;
    }

    switch (dir) {
        case host_to_device:
            copy<T>(buf.device(), buf.host(), buf.size());
            break;
        case device_to_host:
            copy<T>(buf.host(), buf.device(), buf.size());
            break;
    }
}

template<typename T>
void memset(hd_buffer<T> &buf, int ch) {
    std::cout << "xpu::memset: sizeof(T) = "<< sizeof(T) << "; size = " << buf.size() << std::endl;
    std::memset(buf.host(), ch, sizeof(T) * buf.size());
    if (buf.copy_required()) {
        xpu::memset(buf.device(), ch, sizeof(T) * buf.size());
    }
}

template<typename T>
void memset(d_buffer<T> &buf, int ch) {
    xpu::memset(buf.data(), ch, sizeof(T) * buf.size());
}

template<typename T, side S>
struct cmem_io {};

template<typename T, side S>
using cmem_io_t = typename cmem_io<T, S>::type;

template<typename T>
struct cmem_io<T, side::host> {
    using type = hd_buffer<T>;
};

template<typename T>
struct cmem_io<T, side::device> {
    using type = T *;
};

template<typename T, side S>
struct cmem_device {};

template<typename T, side S>
using cmem_device_t = typename cmem_device<T, S>::type;

template<typename T>
struct cmem_device<T, side::host> {
    using type = d_buffer<T>;
};

template<typename T>
struct cmem_device<T, side::device> {
    using type = T *;
};

} // namespace xpu

#endif