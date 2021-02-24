#include "TestKernels.h"

// this is only needed to ensure that kernels without arguments can compile
XPU_KERNEL(TestKernels, empty_kernel, xpu::no_smem) {
}

XPU_KERNEL(TestKernels, vector_add, xpu::no_smem, (const float *) x, (const float *) y, (float *) z, (int) N) {
    int iThread = info.i_block.x * info.n_threads.x + info.i_thread.x;
    if (iThread >= N) {
        return;
    }
    z[iThread] = x[iThread] + y[iThread];
}

XPU_KERNEL(TestKernels, sort_floats, xpu::no_smem, (float *) items, (int) N) {
    xpu::block_sort<float, 32>::storage_t st{};
    xpu::block_sort<float, 32>(st).sort(items, N);
}

XPU_KERNEL(TestKernels, access_cmem, xpu::no_smem, (test_constants *) out) {
    const test_constants &in = xpu::cmem<test_constants>();
    *out = in;
}