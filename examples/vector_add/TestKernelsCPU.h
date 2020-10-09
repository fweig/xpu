#pragma once

#include "TestKernels.h"

class TestKernelsCPU : public TestKernels {
    #define KERNEL_DECL(name, ...) \
        GPUError run_ ## name(GPUKernelParams, ## __VA_ARGS__) override;
    #include "kernels.h"
    #undef KERNEL_DECL
};