#include "bench_device.h"

XPU_IMAGE(bench_device);

XPU_EXPORT(merge<4>);
XPU_EXPORT(merge<8>);
XPU_EXPORT(merge<12>);
XPU_EXPORT(merge<16>);
XPU_EXPORT(merge<32>);
XPU_EXPORT(merge<48>);
XPU_EXPORT(merge<64>);

template<int items_per_thread>
XPU_D void merge<items_per_thread>::operator()(context &ctx, const float *a, const float *b, size_t N, float *c) {
    size_t items_per_block = N;
    size_t offset = items_per_block * ctx.pos().block_idx_x();
    merge_t(ctx.pos(), ctx.smem()).merge(&a[offset], items_per_block, &b[offset], items_per_block, &c[offset * 2], [](float x, float y) { return x < y; });
}

XPU_EXPORT(sort<1>);
XPU_EXPORT(sort<2>);
XPU_EXPORT(sort<4>);
XPU_EXPORT(sort<8>);
XPU_EXPORT(sort<12>);
XPU_EXPORT(sort<16>);
XPU_EXPORT(sort<32>);
XPU_EXPORT(sort<48>);
XPU_EXPORT(sort<64>);

template<int items_per_thread>
XPU_D void sort<items_per_thread>::operator()(context &ctx, float *a, size_t N, float *buf, float **dst) {
    xpu::tpos &pos = ctx.pos();
    size_t items_per_block = N;
    size_t offset = items_per_block * pos.block_idx_x();
    dst[pos.block_idx_x()] = sort_t(pos, ctx.smem()).sort(&a[offset], items_per_block, &buf[offset], [](float x) { return x; });
}
