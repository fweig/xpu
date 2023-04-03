#ifndef XPU_COMMON_IMPL_H
#define XPU_COMMON_IMPL_H

#include "common.h"
#include "detail/buffer_registry.h"

#include <utility>

inline xpu::grid xpu::n_blocks(dim blocks) { return grid{blocks, dim{-1}}; }

inline xpu::grid xpu::n_threads(dim threads) { return grid{dim{-1}, threads}; }

inline xpu::grid::grid(dim b, dim t) : nblocks(b), nthreads(t) {}

inline void xpu::grid::get_compute_grid(dim &block_dim, dim &grid_dim) const {
    if (nblocks.x == -1) {
        grid_dim.x = (nthreads.x + block_dim.x - 1) / block_dim.x;
        grid_dim.y = (nthreads.y > -1 ? (nthreads.y + block_dim.y - 1) / block_dim.y : 1);
        grid_dim.z = (nthreads.z > -1 ? (nthreads.z + block_dim.z - 1) / block_dim.z : 1);
    } else {
        grid_dim.x = std::max(nblocks.x, 1);
        grid_dim.y = std::max(nblocks.y, 1);
        grid_dim.z = std::max(nblocks.z, 1);
    }

    block_dim.y = (grid_dim.y == 1 ? 1 : block_dim.y);
    block_dim.z = (grid_dim.z == 1 ? 1 : block_dim.z);
}

template<typename T>
xpu::buffer<T>::buffer(size_t N, xpu::buffer_type type, T *data) {
    auto &registry = detail::buffer_registry::instance();
    m_data = static_cast<T *>(registry.create(N * sizeof(T), static_cast<detail::buffer_type>(type), data));
}

template<typename T>
XPU_H XPU_D xpu::buffer<T>::~buffer() {
    remove_ref();
}

template<typename T>
XPU_H XPU_D xpu::buffer<T>::buffer(const xpu::buffer<T> &other) {
    m_data = other.m_data;
    add_ref();
}

template<typename T>
XPU_H XPU_D xpu::buffer<T>::buffer(xpu::buffer<T> &&other) {
    m_data = std::exchange(other.m_data, nullptr);
}

template<typename T>
XPU_H XPU_D xpu::buffer<T> &xpu::buffer<T>::operator=(const xpu::buffer<T> &other) {
    if (this != &other) {
        remove_ref();
        m_data = other.m_data;
        add_ref();
    }
    return *this;
}

template<typename T>
XPU_H XPU_D xpu::buffer<T> &xpu::buffer<T>::operator=(xpu::buffer<T> &&other) {
    if (this != &other) {
        remove_ref();
        m_data = std::exchange(other.m_data, nullptr);
    }
    return *this;
}

template<typename T> template<typename U>
XPU_H XPU_D xpu::buffer<T>::buffer(const xpu::buffer<U> &other) {
    m_data = static_cast<T *>(other.get());
    add_ref();
}

template<typename T> template<typename U>
XPU_H XPU_D xpu::buffer<T>::buffer(xpu::buffer<U> &&other) {
    m_data = std::exchange(static_cast<T *>(other.m_data), nullptr);
}

template<typename T> template<typename U>
XPU_H XPU_D xpu::buffer<T> &xpu::buffer<T>::operator=(const xpu::buffer<U> &other) {
    if (this != &other) {
        remove_ref();
        m_data = static_cast<T *>(other.m_data);
        add_ref();
    }
    return *this;
}

template<typename T> template<typename U>
XPU_H XPU_D xpu::buffer<T> &xpu::buffer<T>::operator=(xpu::buffer<U> &&other) {
    if (this != &other) {
        remove_ref();
        m_data = std::exchange(static_cast<T *>(other.m_data), nullptr);
    }
    return *this;
}

template<typename T>
XPU_H XPU_D void xpu::buffer<T>::reset() {
    m_data = nullptr;
    remove_ref();
}

template<typename T>
void xpu::buffer<T>::reset(size_t N, xpu::buffer_type type, T *data) {
    remove_ref();
    auto &registry = detail::buffer_registry::instance();
    m_data = static_cast<T *>(registry.create(N * sizeof(T), static_cast<detail::buffer_type>(type), data));
}

template<typename T>
XPU_H XPU_D void xpu::buffer<T>::add_ref() {
#if !XPU_IS_DEVICE_CODE
    if (m_data != nullptr) {
        detail::buffer_registry::instance().add_ref(m_data);
    }
#endif
}

template<typename T>
XPU_H XPU_D void xpu::buffer<T>::remove_ref() {
#if !XPU_IS_DEVICE_CODE
    if (m_data != nullptr) {
        detail::buffer_registry::instance().remove_ref(m_data);
    }
#endif
}

#endif
