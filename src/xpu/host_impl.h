#ifndef XPU_HOST_IMPL_H
#define XPU_HOST_IMPL_H

#ifndef XPU_HOST_H
#error "xpu/host_impl.h should not be included directly. Include xpu/host.h instead."
#endif

#include "host.h"

template<typename Kernel, typename = typename std::enable_if<xpu::is_kernel<Kernel>::value>::type>
const char *xpu::get_name() {
    return Kernel::get_name();
}

template<typename Kernel, typename = typename std::enable_if<xpu::is_kernel<Kernel>::value>::type, typename... Args>
void xpu::run_kernel(grid params, Args&&... args) {
    std::string backend = "CPU";
    if (active_driver() == driver::cuda) {
        backend = "CUDA";
    }
    std::cout << "Running kernel " << get_name<Kernel>() << " on backend " << backend << std::endl;
    Kernel::dispatch(Kernel::library_type::instance(active_driver()), params, std::forward<Args>(args)...);
}

template<typename DeviceLibrary, typename C>
void xpu::set_cmem(const C &symbol) {
    DeviceLibrary::template cmem<C>::set(DeviceLibrary::instance(active_driver()), symbol);
}

template<typename T>
xpu::hd_buffer<T>::hd_buffer(size_t N) {
    std::cout << "Allocate hd_buffer with " << N << " elems of size " << sizeof(T) << std::endl;
    _size = N;
    hostdata = static_cast<T *>(std::malloc(sizeof(T) * N));

    if (active_driver() == driver::cpu) {
        devicedata = hostdata;
    } else {
        devicedata = device_malloc<T>(N);
    }
}

template<typename T>
xpu::hd_buffer<T>::~hd_buffer() {
    if (hostdata != nullptr) {
        std::free(hostdata);
    }
    if (copy_required()) {
        xpu::free(devicedata);
    }
}

template<typename T>
xpu::hd_buffer<T> &xpu::hd_buffer<T>::operator=(hd_buffer<T> &&other)  {
    _size = other._size;
    hostdata = other.hostdata;
    devicedata = other.devicedata;

    other._size = 0;
    other.hostdata = other.devicedata = nullptr;
    return *this;
}

template<typename T>
xpu::d_buffer<T>::d_buffer(size_t N) {
    _size = N;
    devicedata = device_malloc<T>(N);
}

template<typename T>
xpu::d_buffer<T>::~d_buffer() {
    free(devicedata);
}

template<typename T>
xpu::d_buffer<T> &xpu::d_buffer<T>::operator=(xpu::d_buffer<T> &&other) {
    _size = other._size;
    devicedata = other.devicedata;

    other._size = 0;
    other.devicedata = nullptr;
    return *this;
}

template<typename T>
void xpu::copy(T *dst, const T *src, size_t entries) {
    xpu::memcpy(dst, src, sizeof(T) * entries);
}

template<typename T>
void xpu::copy(hd_buffer<T> &buf, direction dir) {
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
void xpu::memset(hd_buffer<T> &buf, int ch) {
    std::cout << "xpu::memset: sizeof(T) = "<< sizeof(T) << "; size = " << buf.size() << std::endl;
    std::memset(buf.host(), ch, sizeof(T) * buf.size());
    if (buf.copy_required()) {
        xpu::memset(buf.device(), ch, sizeof(T) * buf.size());
    }
}

template<typename T>
void xpu::memset(d_buffer<T> &buf, int ch) {
    xpu::memset(buf.data(), ch, sizeof(T) * buf.size());
}


// Define host specialization for constant memory helpers only here.
// They shouldn't be available for device code.
template<typename T>
struct xpu::cmem_device<T, xpu::side::host> {
    using type = xpu::d_buffer<T>;
};

template<typename T>
struct xpu::cmem_io<T, xpu::side::host> {
    using type = xpu::hd_buffer<T>;
};

#endif
