#include "VectorOps.h"

XPU_IMAGE(VectorOps);

XPU_EXPORT(VectorAdd);
XPU_D void VectorAdd::operator()(context &/*ctx*/, const float * x, const float * y, float * z, size_t N) {
    size_t iThread = xpu::block_idx::x() * xpu::block_dim::x() + xpu::thread_idx::x();
    if (iThread >= N) {
        return;
    }
    z[iThread] = x[iThread] + y[iThread];
}
