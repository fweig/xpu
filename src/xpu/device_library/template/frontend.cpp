#include XPU_DEVICE_LIBRARY_FRONTEND_H
#include <xpu/device.h>

#include <memory>

// #define XPU_DEVICE_LIBRARY_BACKEND_NAME VectorOpsCPU
// #define XPU_DEVICE_LIBRARY_FRONTEND_H "VectorOps.h"

XPU_DEVICE_LIBRARY_NAME &XPU_DEVICE_LIBRARY_NAME::instance(xpu::driver type) {
    static XPU_DEVICE_LIBRARY_BACKEND_NAME testKernelsCPU{};
    static std::unique_ptr<xpu::lib_obj<XPU_DEVICE_LIBRARY_NAME>> testKernelsCUDA{};

    switch (type) {
        case xpu::driver::cpu:
            return testKernelsCPU;
        case xpu::driver::cuda:
            if (testKernelsCUDA == nullptr) {
                // FIXME: Don't hardcode library paths. Add option to set library paths!!!
                testKernelsCUDA.reset(new xpu::lib_obj<XPU_DEVICE_LIBRARY_NAME>("./build/examples/vector_add/lib" XPU_STRINGIZE(XPU_DEVICE_LIBRARY_NAME) "CUDA.so"));
            }
            return *testKernelsCUDA->obj;
    }

    // unreachable
    return testKernelsCPU;
}