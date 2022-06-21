#ifndef XPU_COMMON_IMPL_H
#define XPU_COMMON_IMPL_H

#include "common.h"

inline xpu::grid xpu::grid::n_blocks(dim blocks) { return grid{blocks, dim{-1}}; }

inline xpu::grid xpu::grid::n_threads(dim threads) { return grid{dim{-1}, threads}; }

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
