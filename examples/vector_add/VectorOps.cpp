#include <xpu/kernel.h>

struct NoSHM {};
XPU_KERNEL(add, NoSHM, (const float *) x, (const float *) y, (float *) z, (size_t) N) {
    int iThread = info.i_block.x * info.n_threads.x + info.i_thread.x;
    if (iThread >= N){
        return;
    }
    z[iThread] = x[iThread] + y[iThread];
}