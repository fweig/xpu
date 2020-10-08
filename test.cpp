#include "TestKernels.h"
#include "gpu.h"

#include <algorithm>
#include <iostream>
#include <vector>

int main() {

    // gpu::getBackends()
    // gpu::getDevices()
    // gpu::setDevice()
    gpu::initialize(GPUBackendType::CPU);

    std::vector<float> hx(100);
    std::fill(hx.begin(), hx.end(), 16);
    std::vector<float> hy(100);
    std::fill(hy.begin(), hy.end(), 26);
    std::vector<float> hz(100);

    float *dx = (float *) gpu::malloc(sizeof(float) * 100);
    float *dy = (float *) gpu::malloc(sizeof(float) * 100);
    float *dz = (float *) gpu::malloc(sizeof(float) * 100);

    gpu::memcpy(dx, hx.data(), 100 * sizeof(float));
    gpu::memcpy(dy, hy.data(), 100 * sizeof(float));
    gpu::runKernel(&TestKernels::vectorAdd, GPUKernelParams{.range={100, 0, 0}}, dx, dy, dz);
    gpu::memcpy(hz.data(), dz, 100 * sizeof(float));

    for (auto &x: hz) {
        if (x != 42) {
            std::cout << "Error: result is " << x << std::endl;
            return 1;
        }
    }
    std::cout << "Looking good!" << std::endl;
    return 0;
}
