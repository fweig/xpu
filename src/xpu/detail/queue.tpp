#ifndef XPU_DETAIL_QUEUE_TPP
#define XPU_DETAIL_QUEUE_TPP

#include "../host.h"
#include "log.h"
#include "backend.h"

inline xpu::queue::queue() : m_handle(std::make_shared<detail::queue_handle>()) {
}

inline xpu::queue::queue(xpu::device dev) : m_handle(std::make_shared<detail::queue_handle>(dev.m_impl)) {
}

inline void xpu::queue::copy(const void *from, void *to, size_t size_bytes) {
    if (from == nullptr || to == nullptr) {
        throw std::runtime_error("xpu::queue::copy: invalid pointer");
    }

    if (detail::logger::instance().active()) {
        xpu::ptr_prop src_prop{from};
        xpu::ptr_prop dst_prop{to};

        detail::device_prop src_device;
        detail::backend::call(m_handle->dev.backend, &detail::backend_base::get_properties, &src_device, src_prop.device().device_nr());

        detail::device_prop dst_device;
        detail::backend::call(m_handle->dev.backend, &detail::backend_base::get_properties, &dst_device, dst_prop.device().device_nr());

        XPU_LOG("Copy %lu bytes from %s to %s.", size_bytes, src_device.name.c_str(), dst_device.name.c_str());
    }

    detail::backend::call(m_handle->dev.backend, &detail::backend_base::memcpy_async, to, from, size_bytes, m_handle->handle);
}

template<typename T>
void xpu::queue::copy(buffer<T> buf, xpu::direction dir) {
    buffer_prop props{buf};

    switch (props.type()) {
    case buf_io: {
        T *from = nullptr;
        T *to = nullptr;

        switch (dir) {
        case direction::host_to_device:
            from = props.h_ptr();
            to = props.d_ptr();
            break;
        case direction::device_to_host:
            from = props.d_ptr();
            to = props.h_ptr();
            break;
        }

        if (from == nullptr || to == nullptr) {
            throw std::runtime_error("xpu::queue::copy: invalid buffer");
        }

        copy(from, to, props.size_bytes());
    }
    break;
    default:
        // TODO: for shared buffers, this could become a prefetch
        throw std::runtime_error("xpu::queue::copy: invalid buffer type");
    }
}

inline void xpu::queue::memset(void *dst, int value, size_t size) {
    if (dst == nullptr) {
        throw std::runtime_error("xpu::queue::memset: invalid pointer");
    }

    detail::backend::call(m_handle->dev.backend, &detail::backend_base::memset_async, dst, value, size, m_handle->handle);
}

template<typename T>
void xpu::queue::memset(buffer<T> buf, int value) {
    buffer_prop props{buf};
    T *ptr = props.d_ptr();

    if (ptr == nullptr) {
        throw std::runtime_error("xpu::queue::memset: invalid buffer");
    }

    memset(ptr, value, props.size_bytes());
}

template<typename Kernel, typename... Args>
void xpu::queue::launch(grid params, Args&&... args) {
    detail::runtime::instance().run_kernel<Kernel>(params, m_handle->dev.backend, m_handle->handle, std::forward<Args>(args)...);
}

inline void xpu::queue::wait() {
    detail::backend::call(m_handle->dev.backend, &detail::backend_base::synchronize_queue, m_handle->handle);
}

#endif // XPU_DETAIL_QUEUE_TPP
