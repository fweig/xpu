#ifndef XPU_DRIVER_SYCL_TPOS_IMPL_H
#define XPU_DRIVER_SYCL_TPOS_IMPL_H

#include <sycl/sycl.hpp>

namespace xpu::detail {

class tpos_impl {

public:
    explicit tpos_impl(sycl::nd_item<3> nd_item) : m_nd_item(nd_item) {}

    int thread_idx_x() const { return m_nd_item.get_local_id(0); }
    int thread_idx_y() const { return m_nd_item.get_local_id(1); }
    int thread_idx_z() const { return m_nd_item.get_local_id(2); }

    int block_dim_x() const { return m_nd_item.get_local_range(0); }
    int block_dim_y() const { return m_nd_item.get_local_range(1); }
    int block_dim_z() const { return m_nd_item.get_local_range(2); }

    int block_idx_x() const { return m_nd_item.get_group(0); }
    int block_idx_y() const { return m_nd_item.get_group(1); }
    int block_idx_z() const { return m_nd_item.get_group(2); }

    int grid_dim_x() const { return m_nd_item.get_group_range(0); }
    int grid_dim_y() const { return m_nd_item.get_group_range(1); }
    int grid_dim_z() const { return m_nd_item.get_group_range(2); }

    void barrier() const { sycl::group_barrier(m_nd_item.get_group()); }

    sycl::group<3> group() const { return m_nd_item.get_group(); }

private:
    sycl::nd_item<3> m_nd_item;

};

} // namespace xpu::detail

#endif
