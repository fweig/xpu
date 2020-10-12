#pragma once

#include <xpu/xpu.h>
#include <xpu/device_library/device_library.h>

class XPU_DEVICE_LIBRARY_NAME {

public:
    #define KERNEL_DECL(name, ...) \
        struct name : public xpu::kernel_dispatcher<name, XPU_DEVICE_LIBRARY_NAME> { \
            template<typename... Args> \
            static inline xpu::error dispatch_impl(XPU_DEVICE_LIBRARY_NAME &lib, xpu::grid params, Args &&... args) { \
                return lib.run_ ## name(params, std::forward<Args>(args)...); \
            } \
            static inline const char *name_impl() { \
                return "TestKernels::" #name; \
            } \
        }; \
        virtual xpu::error run_ ## name(xpu::grid, ## __VA_ARGS__) = 0;
    #include XPU_DEVICE_LIBRARY_KERNEL_DEF_FILE 
    #undef KERNEL_DECL

private:
    template<typename, typename, typename...Args>
    friend void xpu::run_kernel(xpu::grid, Args&&...);

    static XPU_DEVICE_LIBRARY_NAME &instance(xpu::driver type);
};

#define KERNEL_DECL(name, ...) \
    template<> \
    struct xpu::is_kernel<XPU_DEVICE_LIBRARY_NAME::name> : std::true_type {};
#include XPU_DEVICE_LIBRARY_KERNEL_DEF_FILE
#undef KERNEL_DECL