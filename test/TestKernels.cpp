#include "TestKernels.h"

XPU_IMAGE(TestKernels);

XPU_CONSTANT(test_constants);

// Ensure that kernels without arguments can compile.
XPU_KERNEL(empty_kernel, xpu::no_smem) {}

XPU_KERNEL(vector_add, xpu::no_smem, const float *x, const float *y, float *z, int N) {
    int iThread = xpu::block_idx::x() * xpu::block_dim::x() + xpu::thread_idx::x();
    if (iThread >= N) {
        return;
    }
    z[iThread] = x[iThread] + y[iThread];
}

using block_sort_t = xpu::block_sort<float, float, 64, 2>;
struct sort_floats_smem {
    using sort_buf_t = typename block_sort_t::storage_t;
    sort_buf_t sortbuf;
};

XPU_KERNEL(sort_float, sort_floats_smem, float *items, int N, float *buf, float **dst) {
    // printf("In sort\n");
    *dst = block_sort_t(smem.sortbuf).sort(items, N, buf, [](const float &x) { return x; });
}

using block_sort_kv_t = xpu::block_sort<unsigned int, key_value_t, 64, 8>;
struct sort_kv_smem {
    using sort_buf_kv_t = typename block_sort_kv_t::storage_t;
    sort_buf_kv_t sortbuf;
};

XPU_KERNEL(sort_struct, sort_kv_smem, key_value_t *items, int N, key_value_t *buf, key_value_t **dst) {
    *dst = block_sort_kv_t(smem.sortbuf).sort(items, N, buf, [](const key_value_t &pair) { return pair.key; });
}

XPU_KERNEL(access_cmem, xpu::no_smem, float3_ *out) {
    if (xpu::thread_idx::x() > 0) {
        return;
    }
    const float3_ &in = xpu::cmem<test_constants>();
    *out = in;
}

XPU_KERNEL(get_thread_idx, xpu::no_smem, int *idx) {
    int iThread = xpu::block_idx::x() * xpu::block_dim::x() + xpu::thread_idx::x();
    idx[iThread] = xpu::thread_idx::x();
}
