#include <xpu/kernel.h>

XPU_KERNEL(add, xpu::no_smem, (const float *) x, (const float *) y, (float *) z, (size_t) N) {
    unsigned int iThread = info.i_block.x * info.n_threads.x + info.i_thread.x;
    if (iThread >= N){
        return;
    }
    z[iThread] = x[iThread] + y[iThread];
}