#include "kernel.h"

#include "TestKernels.h"

struct NoSHM {};

KERNEL_IMPL(vectorAdd, NoSHM, (const float *) x, (const float *) y, (float *) z) {
    z[info.blockIdx.x] = x[info.blockIdx.x] + y[info.blockIdx.x];
}