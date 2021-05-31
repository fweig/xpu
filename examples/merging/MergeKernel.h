#ifndef MERGE_KERNEL_H
#define MERGE_KERNEL_H

#include <xpu/device.h>

struct MergeKernel {};

XPU_EXPORT_KERNEL(MergeKernel, GpuMerge, const float *, size_t, const float *, size_t, float *);

#endif
