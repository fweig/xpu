#ifndef TEST_KERNELS_H
#define TEST_KERNELS_H

#include <xpu/device.h>

struct TestKernels {};

struct float3_ {
    float x;
    float y;
    float z;
};

struct key_value_t {
    unsigned int key;
    unsigned int value;
};

XPU_EXPORT_CONSTANT(TestKernels, float3_, test_constants);

XPU_EXPORT_KERNEL(TestKernels, empty_kernel);
XPU_EXPORT_KERNEL(TestKernels, vector_add, const float *, const float *, float *, int);
XPU_EXPORT_KERNEL(TestKernels, vector_add_timing, const float *, const float *, float *, int);
XPU_EXPORT_KERNEL(TestKernels, sort_float, float *, int, float *, float **);
XPU_EXPORT_KERNEL(TestKernels, sort_struct, key_value_t *, int, key_value_t *, key_value_t **);
XPU_EXPORT_KERNEL(TestKernels, merge, const float *, size_t, const float *, size_t, float *);
XPU_EXPORT_KERNEL(TestKernels, merge_single, const float *, size_t, const float *, size_t, float *);
XPU_EXPORT_KERNEL(TestKernels, access_cmem, float3_ *);
XPU_EXPORT_KERNEL(TestKernels, get_thread_idx, int *);

#endif
