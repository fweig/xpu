#pragma once

#include <xpu/gpu.h>
#include <xpu/device_library/device_library.h>

class XPU_DEVICE_LIBRARY_NAME {

public:
    #define KERNEL_DECL(name, ...) \
        struct name : public kernel_dispatcher<name, XPU_DEVICE_LIBRARY_NAME> { \
            template<typename... Args> \
            static inline GPUError dispatch_impl(XPU_DEVICE_LIBRARY_NAME &lib, GPUKernelParams params, Args &&... args) { \
                return lib.run_ ## name(params, std::forward<Args>(args)...); \
            } \
            static inline const char *name_impl() { \
                return "TestKernels::" #name; \
            } \
        }; \
        virtual GPUError run_ ## name(GPUKernelParams, ## __VA_ARGS__) = 0;
    #include XPU_DEVICE_LIBRARY_KERNEL_DEF_FILE 
    #undef KERNEL_DECL

private:
    template<typename, typename, typename...Args>
    friend void gpu::runKernel(GPUKernelParams, Args&&...);

    static XPU_DEVICE_LIBRARY_NAME &instance(GPUBackendType type);
};

#define KERNEL_DECL(name, ...) \
    template<> \
    struct gpu::is_kernel<XPU_DEVICE_LIBRARY_NAME::name> : std::true_type {};
#include XPU_DEVICE_LIBRARY_KERNEL_DEF_FILE
#undef KERNEL_DECL