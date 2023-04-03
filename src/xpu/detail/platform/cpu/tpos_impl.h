#ifndef XPU_DRIVER_CPU_TPOS_IMPL_H
#define XPU_DRIVER_CPU_TPOS_IMPL_H

#include "this_thread.h"

namespace xpu::detail {

class tpos_impl {

public:
    inline int thread_idx_x() const { return 0; }
    inline int thread_idx_y() const { return 0; }
    inline int thread_idx_z() const { return 0; }

    inline int block_dim_x() const { return 1; }
    inline int block_dim_y() const { return 1; }
    inline int block_dim_z() const { return 1; }

    inline int block_idx_x() const { return detail::this_thread::block_idx.x; }
    inline int block_idx_y() const { return detail::this_thread::block_idx.y; }
    inline int block_idx_z() const { return detail::this_thread::block_idx.z; }

    inline int grid_dim_x() const { return detail::this_thread::grid_dim.x; }
    inline int grid_dim_y() const { return detail::this_thread::grid_dim.y; }
    inline int grid_dim_z() const { return detail::this_thread::grid_dim.z; }
};

} // namespace xpu::detail

#endif