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

XPU_KERNEL(vector_add_timing, xpu::no_smem, const float *x, const float *y, float *z, int N) {
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
    float *res = block_sort_t(smem.sortbuf).sort(items, N, buf, [](const float &x) { return x; });

    if (xpu::block_idx::x() == 0) {
        *dst = res;
    }
}

using block_sort_kv_t = xpu::block_sort<unsigned int, key_value_t, 64, 8>;
struct sort_kv_smem {
    using sort_buf_kv_t = typename block_sort_kv_t::storage_t;
    sort_buf_kv_t sortbuf;
};

XPU_KERNEL(sort_struct, sort_kv_smem, key_value_t *items, int N, key_value_t *buf, key_value_t **dst) {
    key_value_t *res = block_sort_kv_t(smem.sortbuf).sort(items, N, buf, [](const key_value_t &pair) { return pair.key; });

    if (xpu::block_idx::x() == 0) {
        *dst = res;
    }
}

using merge_t = xpu::block_merge<float, 64, 8>;
XPU_KERNEL(merge, merge_t::storage_t, const float *a, size_t size_a, const float *b, size_t size_b, float *dst) {
    merge_t(smem).merge(a, size_a, b, size_b, dst, [](float a, float b) { return a < b; });
}

using merge_single_t = xpu::block_merge<float, 64, 1>;
XPU_KERNEL(merge_single, typename merge_single_t::storage_t, const float *a, size_t size_a, const float *b, size_t size_b, float *dst) {
    merge_single_t(smem).merge(a, size_a, b, size_b, dst, [](float a, float b) { return a < b; });
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
