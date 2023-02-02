#include "MergeKernel.h"

XPU_IMAGE(MergeKernel);

XPU_EXPORT(GpuMerge);
XPU_D void GpuMerge::operator()(context &ctx, const float *a, size_t size_a, const float *b, size_t size_b, float *dst) {
    merge_t(ctx.pos(), ctx.smem()).merge(a, size_a, b, size_b, dst, [](const float &a, const float &b) { return a < b; });
}
