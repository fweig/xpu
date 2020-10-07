#include "TestKernels.h"
#include "gpu.h"

#include <vector>

int main() {

    // gpu::getBackends()
    // gpu::getDevices()
    // gpu::setDevice()
    gpu::initialize(GPUBackendType::CPU);

    std::vector<float> hx(100);
    std::vector<float> hy(100);
    std::vector<float> hz(100);

    float *dx = gpu::malloc(sizeof(float) * 100);
    float *dy = gpu::malloc(sizeof(float) * 100);
    float *dz = gpu::malloc(sizeof(float) * 100);

    gpu::memcpy(dx, hx.data());
    gpu::memcpy(dy, hy.data());
    gpu::runKernel(&TestKernels::vectorAdd, RANGE, dx, dy, dz);
    gpu::memcpy(hz.data(), dz);

    // TODO: Check results...
}
