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

using block_sort_t = xpu::block_sort<float, 64, 1>;
struct sort_floats_smem {
    using sort_buf_t = typename block_sort_t::storage_t;
    sort_buf_t sortbuf;
};

XPU_KERNEL(TestKernels, sort_float, sort_floats_smem, (float *) items, (int) N, (float *) buf, (float **) dst) {
    // printf("In sort\n");
    *dst = block_sort_t(shm.sortbuf).sort(items, N, buf, [](const float &x) { return x; });
}

using block_sort_kv_t = xpu::block_sort<unsigned int, 64, 1>;
struct sort_kv_smem {
    using sort_buf_kv_t = typename block_sort_kv_t::storage_t;
    sort_buf_kv_t sortbuf;
};

XPU_KERNEL(TestKernels, sort_struct, sort_kv_smem, (key_value_t *) items, (int) N, (key_value_t *) buf, (key_value_t **) dst) {
    *dst = block_sort_kv_t(shm.sortbuf).sort(items, N, buf, [](const key_value_t &pair) { return pair.key; });
}

XPU_KERNEL(TestKernels, access_cmem, xpu::no_smem, (test_constants *) out) {
    if (info.i_thread.x > 0) {
        return;
    }
    const test_constants &in = xpu::cmem<test_constants>();
    *out = in;
}

XPU_KERNEL(TestKernels, get_thread_idx, xpu::no_smem, (int *) idx) {
    int iThread = info.i_block.x * info.n_threads.x + info.i_thread.x;
    idx[iThread] = xpu::thread_idx::x();
}
