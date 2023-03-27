#ifndef XPU_HOST_IMPL_H
#define XPU_HOST_IMPL_H

#ifndef XPU_HOST_H
#error "xpu/host_impl.h should not be included directly. Include xpu/host.h instead."
#endif

#include "host.h"

#include "detail/exceptions.h"
#include "detail/runtime.h"
#include "detail/type_info.h"

void xpu::initialize(settings settings) {
    detail::runtime::instance().initialize(settings);
}

inline void *xpu::malloc_host(size_t bytes) {
    return detail::runtime::instance().malloc_host(bytes);
}

template<typename T>
T *xpu::malloc_host(size_t count) {
    return static_cast<T *>(malloc_host(count * sizeof(T)));
}

inline void *xpu::malloc_device(size_t bytes) {
    return detail::runtime::instance().malloc_device(bytes);
}

template<typename T>
T *xpu::malloc_device(size_t count) {
    return static_cast<T *>(malloc_device(count * sizeof(T)));
}

inline void *xpu::malloc_shared(size_t bytes) {
    return detail::runtime::instance().malloc_shared(bytes);
}

template<typename T>
T *xpu::malloc_shared(size_t count) {
    return static_cast<T *>(malloc_shared(count * sizeof(T)));
}

void xpu::free(void *ptr) {
    detail::runtime::instance().free(ptr);
}

void xpu::memcpy(void *dst, const void *src, size_t bytes) {
    return detail::runtime::instance().memcpy(dst, src, bytes);
}

void xpu::memset(void *dst, int ch, size_t bytes) {
    return detail::runtime::instance().memset(dst, ch, bytes);
}

std::vector<xpu::device_prop> xpu::get_devices() {
    return detail::runtime::instance().get_devices();
}

xpu::device_prop xpu::device_properties() {
    return detail::runtime::instance().device_properties();
}

xpu::driver_t xpu::active_driver() {
    return detail::runtime::instance().active_driver();
}

xpu::device_prop xpu::pointer_get_device(const void *ptr) {
    return detail::runtime::instance().pointer_get_device(ptr);
}

template<typename Kernel>
const char *xpu::get_name() {
    return detail::type_name<Kernel>();
}

template<typename Kernel>
std::vector<float> xpu::get_timing() {
    return detail::runtime::instance().get_timing<Kernel>();
}

template<typename Kernel, typename... Args>
void xpu::run_kernel(grid params, Args&&... args) {
    detail::runtime::instance().run_kernel<Kernel>(params, std::forward<Args>(args)...);
}

template<typename Func, typename... Args>
void xpu::call(Args&&... args) {
    detail::runtime::instance().call<Func>(std::forward<Args>(args)...);
}

template<typename C>
void xpu::set_constant(const typename C::data_t &symbol) {
    detail::runtime::instance().set_constant<C>(symbol);
}

template<typename T>
xpu::buffer_prop<T>::buffer_prop(const buffer<T> &buf) {
    detail::buffer_data entry = detail::buffer_registry::instance().get(buf.get());
    m_size_bytes = entry.size;
    m_device = static_cast<T *>(entry.ptr);
    m_host = static_cast<T *>(entry.host_ptr);
    m_type = static_cast<xpu::buffer_type>(entry.type);
}

template<typename T>
xpu::h_view<T>::h_view(buffer<T> &buf) : h_view(buffer_prop{buf}) {}

template<typename T>
xpu::h_view<T>::h_view(const buffer_prop<T> &buf) {
    if (buf.h_ptr() == nullptr) {
        throw std::runtime_error("h_view: buffer not accessible on host");
    }

    m_size = buf.size();
    m_data = buf.h_ptr();
}

template<typename T>
T &xpu::h_view<T>::operator[](size_t i) {
    XPU_CHECK_RANGE("h_view::operator[]", i, m_size);
    return m_data[i];
}

template<typename T>
const T &xpu::h_view<T>::operator[](size_t i) const {
    XPU_CHECK_RANGE("h_view::operator[]", i, m_size);
    return m_data[i];
}

template<typename T>
T &xpu::h_view<T>::at(size_t i) {
    XPU_CHECK_RANGE("h_view::at", i, m_size);
    return m_data[i];
}

template<typename T>
const T &xpu::h_view<T>::at(size_t i) const {
    XPU_CHECK_RANGE("h_view::at", i, m_size);
    return m_data[i];
}

template<typename T>
xpu::hd_buffer<T>::hd_buffer(size_t N) {
    m_size = N;
    m_h = static_cast<T *>(std::malloc(sizeof(T) * N));

    if (active_driver() == cpu) {
        m_d = m_h;
    } else {
        m_d = malloc_device<T>(N);
    }
}

template<typename T>
xpu::hd_buffer<T>::~hd_buffer() {
    reset();
}

template<typename T>
xpu::hd_buffer<T> &xpu::hd_buffer<T>::operator=(hd_buffer<T> &&other)  {
    if (this == &other) {
        return *this;
    }
    reset();

    m_size = other.m_size;
    m_h = other.m_h;
    m_d = other.m_d;

    other.m_size = 0;
    other.m_h = other.m_d = nullptr;
    return *this;
}

template<typename T>
void xpu::hd_buffer<T>::reset() {
    std::free(m_h);
    if (copy_required()) {
        xpu::free(m_d);
    }
    m_h = m_d = nullptr;
    m_size = 0;
}

template<typename T>
xpu::d_buffer<T>::d_buffer(size_t N) {
    m_size = N;
    m_d = malloc_device<T>(N);
}

template<typename T>
xpu::d_buffer<T>::~d_buffer() {
    reset();
}

template<typename T>
xpu::d_buffer<T> &xpu::d_buffer<T>::operator=(xpu::d_buffer<T> &&other) {
    reset();

    m_size = other.m_size;
    m_d = other.m_d;

    other.m_size = 0;
    other.m_d = nullptr;
    return *this;
}

template<typename T>
void xpu::d_buffer<T>::reset() {
    xpu::free(m_d);
    m_d = nullptr;
    m_size = 0;
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
            copy<T>(buf.d(), buf.h(), buf.size());
            break;
        case device_to_host:
            copy<T>(buf.h(), buf.d(), buf.size());
            break;
    }
}

template<typename T>
void xpu::copy(buffer<T> &buf, direction dir) {
    detail::buffer_data entry = detail::buffer_registry::instance().get(buf.get());

    if (entry.type != detail::io_buffer) {
        throw std::runtime_error("Buffer is not an IO buffer.");
    }

    if (entry.ptr == entry.host_ptr) {
        return;
    }

    void *dst = nullptr;
    void *src = nullptr;

    switch (dir) {
        case host_to_device:
            dst = entry.ptr;
            src = entry.host_ptr;
            break;
        case device_to_host:
            dst = entry.host_ptr;
            src = entry.ptr;
            break;
    }

    xpu::memcpy(dst, src, entry.size);
}

template<typename T>
void xpu::memset(hd_buffer<T> &buf, int ch) {
    std::memset(buf.h(), ch, sizeof(T) * buf.size());
    if (buf.copy_required()) {
        xpu::memset(buf.d(), ch, sizeof(T) * buf.size());
    }
}

template<typename T>
void xpu::memset(d_buffer<T> &buf, int ch) {
    xpu::memset(buf.d(), ch, sizeof(T) * buf.size());
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
