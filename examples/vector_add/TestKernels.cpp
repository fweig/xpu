#include "TestKernels.h"

TestKernels &TestKernels::instance() {
    static CPUTestKernels theInstance{};
    return theInstance;
}

#define KERNEL_DECL(name, ...) \
    template<> \
    const char *gpu::getName<TestKernels::name>() { return "TestKernels::" #name; }
#include "testKernels.h"
#undef DECL_KERNEL