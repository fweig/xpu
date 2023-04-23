#ifndef XPU_DETAIL_QUEUE_TPP
#define XPU_DETAIL_QUEUE_TPP

#include "../host.h"
#include "log.h"
#include "backend.h"
#include "config.h"
#include "timers.h"

inline xpu::queue::queue() : m_handle(std::make_shared<detail::queue_handle>()) {
}

inline xpu::queue::queue(xpu::device dev) : m_handle(std::make_shared<detail::queue_handle>(dev.m_impl)) {
}

inline void xpu::queue::copy(const void *from, void *to, size_t size_bytes) {
    if (from == nullptr || to == nullptr) {
        throw std::runtime_error("xpu::queue::copy: invalid pointer");
    }

    if (!detail::config::profile) {
        do_copy(from, to, size_bytes, nullptr);
    } else {
        double ms;
        do_copy(from, to, size_bytes, &ms);

        ptr_prop src_prop{from};
        detail::direction_t dir = src_prop.type() == xpu::mem_type::host ? detail::dir_h2d : detail::dir_d2h;
        detail::add_memcpy_time(ms, dir);
    }
}

template<typename T>
void xpu::queue::copy(buffer<T> buf, xpu::direction dir) {
    buffer_prop props{buf};

    switch (props.type()) {
    case buf_io: {
        T *from = nullptr;
        T *to = nullptr;

        switch (dir) {
        case h2d:
            from = props.h_ptr();
            to = props.d_ptr();
            break;
        case d2h:
            from = props.d_ptr();
            to = props.h_ptr();
            break;
        }

        if (from == nullptr || to == nullptr) {
            throw std::runtime_error("xpu::queue::copy: invalid buffer");
        }

        log_copy(from, to, props.size_bytes());

        if (!detail::config::profile) {
            do_copy(from, to, props.size_bytes(), nullptr);
        } else {
            double ms;
            do_copy(from, to, props.size_bytes(), &ms);
            detail::add_memcpy_time(ms, static_cast<detail::direction_t>(dir));
        }
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

    if (!detail::config::profile) {
        detail::backend::call(m_handle->dev.backend, &detail::backend_base::memset_async, dst, value, size, m_handle->handle, nullptr);
    } else {
        double ms;
        detail::backend::call(m_handle->dev.backend, &detail::backend_base::memset_async, dst, value, size, m_handle->handle, &ms);
        detail::add_memset_time(ms);
    }
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

inline void xpu::queue::do_copy(const void *from, void *to, size_t size, double *ms) {
    detail::backend::call(m_handle->dev.backend, &detail::backend_base::memcpy_async,
            to, from, size, m_handle->handle, ms);
}

inline void xpu::queue::log_copy(const void *from, const void *to, size_t size) {
    if (!detail::config::logging) {
        return;
    }

    ptr_prop src_prop{from};
    ptr_prop dst_prop{to};

    detail::device_prop src_device;
    detail::backend::call(m_handle->dev.backend, &detail::backend_base::get_properties, &src_device, src_prop.device().device_nr());

    detail::device_prop dst_device;
    detail::backend::call(m_handle->dev.backend, &detail::backend_base::get_properties, &dst_device, dst_prop.device().device_nr());

    XPU_LOG("Copy %lu bytes from %s to %s.", size, src_device.name.c_str(), dst_device.name.c_str());
}

#endif // XPU_DETAIL_QUEUE_TPP
