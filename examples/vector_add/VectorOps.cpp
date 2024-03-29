#include "VectorOps.h"

XPU_IMAGE(VectorOps);

XPU_EXPORT(VectorAdd);
XPU_D void VectorAdd::operator()(context &ctx, xpu::buffer<const float> x, xpu::buffer<const float> y, xpu::buffer<float> z, size_t N) {
    xpu::tpos &pos = ctx.pos();
    size_t iThread = pos.block_idx_x() * pos.block_dim_x() + pos.thread_idx_x();
    if (iThread >= N) {
        return;
    }
    z[iThread] = x[iThread] + y[iThread];
}
