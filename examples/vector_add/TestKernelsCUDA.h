#pragma once

#include "TestKernels.h"

class TestKernelsCUDA : public TestKernels {
    #define KERNEL_DECL(name, ...) \
        GPUError run_ ## name(GPUKernelParams, ## __VA_ARGS__) override;
    #include "kernels.h"
    #undef KERNEL_DECL
};