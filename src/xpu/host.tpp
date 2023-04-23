#ifndef XPU_HOST_IMPL_H
#define XPU_HOST_IMPL_H

#ifndef XPU_HOST_H
#error "xpu/host_impl.h should not be included directly. Include xpu/host.h instead."
#endif

#include "host.h"

#include "detail/exceptions.h"
#include "detail/runtime.h"
#include "detail/timers.h"
#include "detail/type_info.h"

#include "detail/queue.tpp"

void xpu::initialize(settings settings) {
    detail::runtime::instance().initialize(settings);
}

template<typename I>
inline void xpu::preload() {
    detail::runtime::instance().preload_image<I>();
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

inline void xpu::stack_alloc(size_t bytes) {
    detail::buffer_registry::instance().stack_alloc(
        detail::runtime::instance().active_device(),
        bytes
    );
}

inline void xpu::stack_pop(void *head) {
    detail::buffer_registry::instance().stack_pop(
        detail::runtime::instance().active_device(),
        head
    );
}

inline std::vector<xpu::device> xpu::device::all() {
    auto dev_impl = detail::runtime::instance().get_devices();

    std::vector<xpu::device> devices;
    devices.reserve(dev_impl.size());

    for (auto &d : dev_impl) {
        devices.push_back(device{d});
    }

    return devices;
}

inline xpu::device xpu::device::active() {
    return device{detail::runtime::instance().active_device()};
}

inline xpu::device::device() {
    m_impl = detail::runtime::instance().get_devices()[0];
}

inline xpu::device::device(std::string_view xpuid) {
    m_impl = detail::runtime::instance().get_device(xpuid);
}

inline xpu::device::device(driver_t backend, int device) {
    m_impl = detail::runtime::instance().get_device(static_cast<detail::driver_t>(backend), device);
}

inline xpu::device::device(int id) {
    m_impl = detail::runtime::instance().get_device(id);
}

inline xpu::device_prop::device_prop(xpu::device dev) {
    m_prop = detail::runtime::instance().device_properties(dev.id());
}



template<typename Kernel>
const char *xpu::get_name() {
    return detail::type_name<Kernel>();
}

template<typename Kernel, typename... Args>
void xpu::run_kernel(grid params, Args&&... args) {
    detail::device dev = detail::runtime::instance().active_device();
    detail::runtime::instance().run_kernel<Kernel>(params, dev.backend, nullptr, std::forward<Args>(args)...);
}

template<typename Func, typename... Args>
void xpu::call(Args&&... args) {
    detail::runtime::instance().call<Func>(std::forward<Args>(args)...);
}

template<typename C>
void xpu::set_constant(const typename C::data_t &symbol) {
    detail::runtime::instance().set_constant<C>(symbol);
}

inline xpu::ptr_prop::ptr_prop(const void *ptr) {
    detail::runtime::instance().get_ptr_prop(ptr, &m_prop);
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

inline void xpu::push_timer(std::string_view name) {
    detail::push_timer(name);
}

inline xpu::timings xpu::pop_timer() {
    return timings{detail::pop_timer()};
}

inline std::vector<xpu::kernel_timings> xpu::timings::kernels() const {
    std::vector<kernel_timings> kernels;
    kernels.reserve(m_t.kernels.size());
    for (auto &k : m_t.kernels) {
        kernels.emplace_back(k);
    }
    return kernels;
}

inline std::vector<xpu::timings> xpu::timings::children() const {
    std::vector<timings> children;
    children.reserve(m_t.children.size());
    for (auto &c : m_t.children) {
        children.emplace_back(c);
    }
    return children;
}

inline xpu::kernel_timings xpu::timings::kernel(std::string_view name) const {
    auto it = std::find_if(m_t.kernels.begin(), m_t.kernels.end(), [&](const auto &k) {
        return k.name == name;
    });
    if (it == m_t.kernels.end()) {
        detail::kernel_timings kt;
        kt.name = name;
        return kernel_timings{kt};
    }
    return kernel_timings{*it};
}

template<typename T>
void xpu::copy(T *dst, const T *src, size_t entries) {
    xpu::memcpy(dst, src, sizeof(T) * entries);
}

template<typename T>
void xpu::copy(buffer<T> &buf, direction dir) {
    detail::buffer_data entry = detail::buffer_registry::instance().get(buf.get());

    if (entry.type != detail::buf_io) {
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

#endif
