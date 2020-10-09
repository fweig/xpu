#pragma once

#include <xpu/gpu.h>

class TestKernels {

public:
    #define KERNEL_DECL(name, ...) \
        struct name : gpu::Kernel<TestKernels> { \
            template<typename... Args> \
            static GPUError dispatch(TestKernels &lib, GPUKernelParams params, Args &&... args) { \
                return lib.run_ ## name(params, std::forward<Args>(args)...); \
            } \
        }; \
        virtual GPUError run_ ## name(GPUKernelParams, ## __VA_ARGS__) = 0;
    #include "testKernels.h"
    #undef KERNEL_DECL

private:
    template<typename, typename, typename...Args>
    friend void gpu::runKernel(GPUKernelParams, Args&&...);

    static TestKernels &instance();
};

class CPUTestKernels : public TestKernels {
    #define KERNEL_DECL(name, ...) \
        GPUError run_ ## name(GPUKernelParams, ## __VA_ARGS__) override;
    #include "testKernels.h"
    #undef KERNEL_DECL
};

#define KERNEL_DECL(name, ...) \
    template<> \
    struct gpu::is_kernel<TestKernels::name> : std::true_type {};
#include "testKernels.h"
#undef KERNEL_DECL