#include <xpu/device.h>

XPU_KERNEL(vector_add, xpu::no_smem, (const float *) x, (const float *) y, (float *) z, (int) N) {
    int iThread = info.i_block.x * info.n_threads.x + info.i_thread.x;
    if (iThread >= N) {
        return;
    }
    z[iThread] = x[iThread] + y[iThread];
}

XPU_KERNEL(sort_floats, xpu::no_smem, (float *) items, (int) N) {
    xpu::block_sort<float, 32>::storage_t st{};
    xpu::block_sort<float, 32>(st).sort(items, N);
}