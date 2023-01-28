#ifndef MERGE_KERNEL_H
#define MERGE_KERNEL_H

#include <xpu/device.h>

struct MergeKernel {};

struct GpuMerge : xpu::kernel<MergeKernel> {
    using merge_t = xpu::block_merge<float, 64, 4>;
    using shared_memory = merge_t::storage_t;
    using context = xpu::kernel_context<shared_memory>;
    XPU_D void operator()(context &, const float *, size_t, const float *, size_t, float *);
};

#endif
