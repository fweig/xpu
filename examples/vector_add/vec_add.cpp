#include <xpu/kernel.h>

#include "TestKernels.h"

struct NoSHM {};


KERNEL_IMPL(vectorAdd, NoSHM, (const float *) x, (const float *) y, (float *) z, (size_t) N) {
    // z[iThread(dim::x)]
    // z[iBlock()]
    // z[nBlocks()]
    // z[nThreads()]
    // int idx = xpu::iBlock() * xpu::nThreads() + xpu::iThread();  
    int iThread = info.blockIdx.x * info.nThreads.x + info.threadIdx.x;
    if (iThread >= N) return;
    z[iThread] = x[iThread] + y[iThread];
}