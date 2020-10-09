#include "TestKernels.h"
#include "TestKernelsCPU.h"
#include <xpu/dl_utils.h>

#include <memory>

TestKernels &TestKernels::instance(GPUBackendType type) {
    static TestKernelsCPU testKernelsCPU{};
    static std::unique_ptr<LibObj<TestKernels>> testKernelsCUDA{};

    switch (type) {
        case GPUBackendType::CPU:
            return testKernelsCPU;
        case GPUBackendType::CUDA:
            if (testKernelsCUDA == nullptr) {
                testKernelsCUDA.reset(new LibObj<TestKernels>("./build/libTestKernelsCUDA.so"));
            }
            return *testKernelsCUDA->obj;
    }
}

#define KERNEL_DECL(name, ...) \
    template<> \
    const char *gpu::getName<TestKernels::name>() { return "TestKernels::" #name; }
#include "kernels.h"
#undef DECL_KERNEL