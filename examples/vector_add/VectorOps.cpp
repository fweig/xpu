#include "VectorOps.h"

XPU_IMAGE(VectorOps);

XPU_KERNEL(VectorAdd, xpu::no_smem, const float * x, const float * y, float * z, size_t N) {
    unsigned int iThread = xpu::block_idx::x() * xpu::block_dim::x() + xpu::thread_idx::x();
    if (iThread >= N) {
        return;
    }
    z[iThread] = x[iThread] + y[iThread];
}