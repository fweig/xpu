#ifndef XPU_DRIVER_HIP_CUDA_TPOS_IMPL_H
#define XPU_DRIVER_HIP_CUDA_TPOS_IMPL_H

#include "../../defines.h"

#if XPU_IS_CUDA
#define XPU_CHOOSE(hip, cuda) cuda
#elif XPU_IS_HIP
#define XPU_CHOOSE(hip, cuda) hip
#else
#error "Unsupported unsupported target"
#endif

namespace xpu::detail {

class tpos_impl {

public:
    XPU_D int thread_idx_x() const { return XPU_CHOOSE(hipThreadIdx_x, threadIdx.x); }
    XPU_D int thread_idx_y() const { return XPU_CHOOSE(hipThreadIdx_y, threadIdx.y); }
    XPU_D int thread_idx_z() const { return XPU_CHOOSE(hipThreadIdx_z, threadIdx.z); }

    XPU_D int block_dim_x() const { return XPU_CHOOSE(hipBlockDim_x, blockDim.x); }
    XPU_D int block_dim_y() const { return XPU_CHOOSE(hipBlockDim_y, blockDim.y); }
    XPU_D int block_dim_z() const { return XPU_CHOOSE(hipBlockDim_z, blockDim.z); }

    XPU_D int block_idx_x() const { return XPU_CHOOSE(hipBlockIdx_x, blockIdx.x); }
    XPU_D int block_idx_y() const { return XPU_CHOOSE(hipBlockIdx_y, blockIdx.y); }
    XPU_D int block_idx_z() const { return XPU_CHOOSE(hipBlockIdx_z, blockIdx.z); }

    XPU_D int grid_dim_x() const { return XPU_CHOOSE(hipGridDim_x, gridDim.x); }
    XPU_D int grid_dim_y() const { return XPU_CHOOSE(hipGridDim_y, gridDim.y); }
    XPU_D int grid_dim_z() const { return XPU_CHOOSE(hipGridDim_z, gridDim.z); }
};

} // namespace xpu::detail

#endif
