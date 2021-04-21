#include "mergeSortOps.h"


using block_sort_t = xpu::block_sort<unsigned int, 64, 1>;
struct sort_floats_smem {
    using sort_buf_t = typename block_sort_t::storage_t;
    sort_buf_t sortbuf;
};


// this is only needed to ensure that kernels without arguments can compile
XPU_KERNEL(mergeSortOps, empty_kernel, xpu::no_smem) {
}


XPU_KERNEL(mergeSortOps, sort, sort_floats_smem, (unsigned int *) items, (int) N, (unsigned int *) buf, (unsigned int **) dst) {
    // printf("In sort\n");
    *dst = block_sort_t(shm.sortbuf).sort(items, N, buf);
}

using block_sort_kv_t = xpu::block_sort<key_value_t, 64, 1>;
struct sort_kv_smem {
    using sort_buf_kv_t = typename block_sort_kv_t::storage_t;
    sort_buf_kv_t sortbuf;
};

XPU_KERNEL(mergeSortOps, sort_struct, sort_kv_smem, (key_value_t *) items, (int) N, (key_value_t *) buf, (key_value_t **) dst) {
    *dst = block_sort_kv_t(shm.sortbuf).sort(items, N, buf);
}