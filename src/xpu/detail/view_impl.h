#ifndef XPU_DETAIL_VIEW_IMPL_H
#define XPU_DETAIL_VIEW_IMPL_H

#ifndef XPU_DEVICE_H
#error "This file should not be included directly. Include xpu/device.h instead."
#endif

template<typename T>
XPU_D xpu::view<T>::view(T *data, size_t size) : m_data(data), m_size(size) {}

template<typename T>
XPU_D xpu::view<T>::view(xpu::buffer<T> &buf, size_t size) : m_data(buf.data()), m_size(size) {}

template<typename T>
XPU_D T &xpu::view<T>::operator[](size_t i) {
    XPU_ASSERT(i < m_size);
    return m_data[i];
}

template<typename T>
XPU_D const T &xpu::view<T>::operator[](size_t i) const {
    XPU_ASSERT(i < m_size);
    return m_data[i];
}

template<typename T>
XPU_D T &xpu::view<T>::at(size_t i) {
    XPU_ASSERT(i < m_size);
    return m_data[i];
}

template<typename T>
XPU_D const T &xpu::view<T>::at(size_t i) const {
    XPU_ASSERT(i < m_size);
    return m_data[i];
}

#endif // XPU_DETAIL_VIEW_IMPL_H
