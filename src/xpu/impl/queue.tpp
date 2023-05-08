#ifndef XPU_DETAIL_QUEUE_TPP
#define XPU_DETAIL_QUEUE_TPP

#include "../host.h"
#include "../detail/log.h"
#include "../detail/backend.h"
#include "../detail/config.h"
#include "../detail/timers.h"

inline xpu::queue::queue() : m_handle(std::make_shared<detail::queue_handle>()) {
}

inline xpu::queue::queue(xpu::device dev) : m_handle(std::make_shared<detail::queue_handle>(dev.impl())) {
}

inline void xpu::queue::memcpy(void *dst, const void *src,  size_t size_bytes) {
    if (size_bytes == 0) {
        return;
    }
    if (dst == nullptr) {
        detail::throw_invalid_argument("xpu::queue::memcpy", "dst is null");
    }
    if (src == nullptr) {
        detail::throw_invalid_argument("xpu::queue::memcpy", "src is null");
    }

    if (!detail::config::profile) {
        do_copy(src, dst, size_bytes, nullptr);
    } else {
        double ms;
        do_copy(src, dst, size_bytes, &ms);

        ptr_prop src_prop{src};
        detail::direction_t dir = src_prop.is_host() ? detail::dir_h2d : detail::dir_d2h;
        detail::add_memcpy_time(ms, dir, size_bytes);
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
            from = props.h_ptr<T>();
            to = props.d_ptr<T>();
            break;
        case d2h:
            from = props.d_ptr<T>();
            to = props.h_ptr<T>();
            break;
        }

        if (from == nullptr || to == nullptr) {
            throw std::runtime_error("xpu::queue::copy: invalid buffer");
        }

        if (from == to) {
            return;
        }

        log_copy(from, to, props.size_bytes());

        if (!detail::config::profile) {
            do_copy(from, to, props.size_bytes(), nullptr);
        } else {
            double ms;
            do_copy(from, to, props.size_bytes(), &ms);
            detail::add_memcpy_time(ms, static_cast<detail::direction_t>(dir), props.size_bytes());
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
        detail::add_memset_time(ms, size);
    }
}

template<typename T>
void xpu::queue::memset(buffer<T> buf, int value) {
    buffer_prop props{buf};
    T *ptr = props.d_ptr<T>();

    if (ptr == nullptr) {
        throw std::runtime_error("xpu::queue::memset: invalid buffer");
    }

    memset(ptr, value, props.size_bytes());
}

template<typename Kernel, typename... Args>
void xpu::queue::launch(grid params, Args&&... args) {
    static_assert(detail::is_kernel_v<Kernel>, "xpu::queue::launch: invalid kernel type");
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
