#include "bench_device.h"

XPU_IMAGE(bench_device);

#if XPU_IS_CUDA
#define MERGE_BLOCK_SIZE 32
#else
#define MERGE_BLOCK_SIZE 64
#endif

#define MERGE_KERNEL(name, elems_per_thread) \
    XPU_BLOCK_SIZE_1D(name, MERGE_BLOCK_SIZE); \
    using block_merge ## elems_per_thread = xpu::block_merge<float, xpu::block_size<name>::value.x, elems_per_thread>; \
    XPU_KERNEL(name, block_merge ## elems_per_thread::storage_t, const float *a, const float *b, size_t N, float *c) { \
        size_t items_per_block = N; \
        size_t offset = items_per_block * xpu::block_idx::x(); \
        block_merge ## elems_per_thread(smem).merge(&a[offset], items_per_block, &b[offset], items_per_block, &c[offset * 2], [](float x, float y) { return x < y; }); \
    }

#define SORT_KERNEL(name, elems_per_thread) \
    XPU_BLOCK_SIZE_1D(name, MERGE_BLOCK_SIZE); \
    using block_sort ## elems_per_thread = xpu::block_sort<float, float, xpu::block_size<name>::value.x, elems_per_thread>; \
    XPU_KERNEL(name, block_sort ## elems_per_thread::storage_t, float *a, size_t N, float *buf, float **dst) { \
        size_t items_per_block = N; \
        size_t offset = items_per_block * xpu::block_idx::x(); \
        dst[xpu::block_idx::x()] = block_sort ## elems_per_thread(smem).sort(&a[offset], items_per_block, &buf[offset], [](float x) { return x; }); \
    }

MERGE_KERNEL(merge_i4, 4);
MERGE_KERNEL(merge_i8, 8);
MERGE_KERNEL(merge_i12, 12);
MERGE_KERNEL(merge_i16, 16);
MERGE_KERNEL(merge_i32, 32);
MERGE_KERNEL(merge_i48, 48);
MERGE_KERNEL(merge_i64, 64);

SORT_KERNEL(sort_i1, 1);
SORT_KERNEL(sort_i2, 2);
SORT_KERNEL(sort_i4, 4);
SORT_KERNEL(sort_i8, 8);
SORT_KERNEL(sort_i12, 12);
SORT_KERNEL(sort_i16, 16);
SORT_KERNEL(sort_i32, 32);
SORT_KERNEL(sort_i48, 48);
SORT_KERNEL(sort_i64, 64);
