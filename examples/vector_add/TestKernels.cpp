#include "TestKernels.h"

TestKernels &TestKernels::instance() {
    static CPUTestKernels theInstance{};
    return theInstance;
}