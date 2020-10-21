#pragma once

#include XPU_DEVICE_LIBRARY_FRONTEND_H

class XPU_DEVICE_LIBRARY_BACKEND_NAME : public XPU_DEVICE_LIBRARY_NAME {
    #define XPU_KERNEL_DECL(name, ...) \
        xpu::error run_ ## name(xpu::grid, ## __VA_ARGS__) override;
    #include XPU_DEVICE_LIBRARY_KERNEL_DEF_FILE
    #undef XPU_KERNEL_DECL
};