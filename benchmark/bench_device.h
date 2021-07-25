#ifndef XPU_BENCH_DEVICE_H
#define XPU_BENCH_DEVICE_H

#include <xpu/device.h>

struct bench_device {};

XPU_EXPORT_KERNEL(bench_device, merge_i4, const float *, const float *, size_t N, float *);
XPU_EXPORT_KERNEL(bench_device, merge_i8, const float *, const float *, size_t N, float *);
XPU_EXPORT_KERNEL(bench_device, merge_i12, const float *, const float *, size_t N, float *);
XPU_EXPORT_KERNEL(bench_device, merge_i16, const float *, const float *, size_t N, float *);
XPU_EXPORT_KERNEL(bench_device, merge_i32, const float *, const float *, size_t N, float *);
XPU_EXPORT_KERNEL(bench_device, merge_i48, const float *, const float *, size_t N, float *);
XPU_EXPORT_KERNEL(bench_device, merge_i64, const float *, const float *, size_t N, float *);

XPU_EXPORT_KERNEL(bench_device, sort_i1, float *, size_t, float *, float **);
XPU_EXPORT_KERNEL(bench_device, sort_i2, float *, size_t, float *, float **);
XPU_EXPORT_KERNEL(bench_device, sort_i4, float *, size_t, float *, float **);
XPU_EXPORT_KERNEL(bench_device, sort_i8, float *, size_t, float *, float **);
XPU_EXPORT_KERNEL(bench_device, sort_i12, float *, size_t, float *, float **);
XPU_EXPORT_KERNEL(bench_device, sort_i16, float *, size_t, float *, float **);
XPU_EXPORT_KERNEL(bench_device, sort_i32, float *, size_t, float *, float **);
XPU_EXPORT_KERNEL(bench_device, sort_i48, float *, size_t, float *, float **);
XPU_EXPORT_KERNEL(bench_device, sort_i64, float *, size_t, float *, float **);

#endif
