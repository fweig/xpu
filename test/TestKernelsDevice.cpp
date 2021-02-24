#include "TestKernels.h"

// this is only needed to ensure that kernels without arguments can compile
XPU_KERNEL(TestKernels, empty_kernel, xpu::no_smem) {
}

XPU_KERNEL(TestKernels, vector_add, xpu::no_smem, (const float *) x, (const float *) y, (float *) z, (int) N) {
    printf("In Kernel vector_add\n");
    int iThread = info.i_block.x * info.n_threads.x + info.i_thread.x;
    if (iThread >= N) {
        return;
    }
    z[iThread] = x[iThread] + y[iThread];
}

XPU_KERNEL(TestKernels, sort_floats, xpu::no_smem, (float *) items, (int) N) {
    xpu::block_sort<float, 64>::storage_t st{};
    // printf("In sort\n");
    xpu::block_sort<float, 64>(st).sort(items, N);
}

XPU_KERNEL(TestKernels, access_cmem, xpu::no_smem, (test_constants *) out) {
    if (info.i_thread.x > 0) {
        return;
    }
    const test_constants &in = xpu::cmem<test_constants>();
    *out = in;
}

XPU_KERNEL(TestKernels, get_thread_idx, xpu::no_smem, (int *) idx) {
    idx[info.i_thread.x] = xpu::thread_idx::x();
}