#pragma once

#include XPU_DEVICE_LIBRARY_FRONTEND_H

class XPU_DEVICE_LIBRARY_BACKEND_NAME : public XPU_DEVICE_LIBRARY_NAME {
    #define KERNEL_DECL(name, ...) \
        GPUError run_ ## name(GPUKernelParams, ## __VA_ARGS__) override;
    #include XPU_DEVICE_LIBRARY_KERNEL_DEF_FILE
    #undef KERNEL_DECL
};