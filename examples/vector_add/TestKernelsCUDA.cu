#include "TestKernelsCUDA.h"

#include "vec_add.cpp"

extern "C" TestKernels *create() {
    return new TestKernelsCUDA{};
}

extern "C" void destroy(TestKernelsCUDA *obj) {
    delete obj;
}