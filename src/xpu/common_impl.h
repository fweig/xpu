#ifndef XPU_COMMON_IMPL_H
#define XPU_COMMON_IMPL_H

#include "common.h"

inline xpu::grid xpu::grid::n_blocks(dim blocks) { return grid{blocks, dim{-1}}; }

inline xpu::grid xpu::grid::n_threads(dim threads) { return grid{dim{-1}, threads}; }

inline xpu::grid::grid(dim b, dim t) : blocks(b), threads(t) {}

// Always provide the device side for constant memory classes
// host side is provided in host_impl.h
template<typename T>
struct xpu::cmem_io<T, xpu::side::device> {
    using type = T *;
};

template<typename T>
struct xpu::cmem_device<T, xpu::side::device> {
    using type = T *;
};

#endif
