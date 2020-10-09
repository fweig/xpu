#include <xpu/kernel.h>

#include "TestKernels.h"

struct NoSHM {};

KERNEL_IMPL(vectorAdd, NoSHM, (const float *) x, (const float *) y, (float *) z) {
    // z[iThread(dim::x)]
    // z[iBlock()]
    // z[nBlocks()]
    // z[nThreads()]
    z[info.blockIdx.x] = x[info.blockIdx.x] + y[info.blockIdx.x];
}