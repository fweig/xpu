#ifndef TEST_KERNELS_H
#define TEST_KERNELS_H

#define XPU_DEVICE_LIBRARY test_kernels

#include <xpu/device.h>

struct test_constants {
    float x;
    float y;
    float z;
};

#define XPU_KERNEL_DECL_DEF <test_kernels.def>
#include <xpu/device_library_h.def>
#undef XPU_KERNEL_DECL_DEF

#endif