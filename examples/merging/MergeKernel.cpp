#include "MergeKernel.h"

XPU_IMAGE(MergeKernel);

using merge_t = xpu::block_merge<float, 64, 4>;

struct GpuMergeSMem {
    typename merge_t::storage_t merge_buf;
};

XPU_KERNEL(GpuMerge, GpuMergeSMem, const float *a, size_t size_a, const float *b, size_t size_b, float *dst) {
    merge_t(smem.merge_buf).merge(a, size_a, b, size_b, dst, [](const float &a, const float &b) { return a < b; });
}
