#pragma once

#include "gpu.h"

class TestKernels {

public:
    static TestKernels &instance();

    #define KERNEL_DECL(name, ...) \
        virtual GPUError name(gpu::internal::KernelTag, GPUKernelParams, ## __VA_ARGS__) = 0;
    #include "testKernels.h"
    #undef KERNEL_DECL
};

class CPUTestKernels : public TestKernels {
    #define KERNEL_DECL(name, ...) \
        GPUError name(gpu::internal::KernelTag, GPUKernelParams, ## __VA_ARGS__) override;
    #include "testKernels.h"
    #undef KERNEL_DECL
};