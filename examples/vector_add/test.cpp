#include "TestKernels.h"
#include <xpu/gpu.h>

#include <algorithm>
#include <iostream>
#include <vector>

int main() {

    // gpu::getBackends()
    // gpu::getDevices()
    // gpu::setDevice()
    // xpu::initialize(xpu::backend::cpu)
    gpu::initialize(GPUBackendType::CUDA); // or GPUBackendType::CUDA or GPUBackendType::HIP

    std::vector<float> hx(100);
    std::fill(hx.begin(), hx.end(), 16);
    std::vector<float> hy(100);
    std::fill(hy.begin(), hy.end(), 26);
    std::vector<float> hz(100);

    // gpu::d_alloc<float>(100)
    // gpu::io_alloc<float>(100)
    // gpu::d_ptr<float> dx = gpu::alloc_d<float>(100)
    // gpu::hd_ptr<float> dy = gpu::alloc_io<float>(100)
    // gpu::DAlloc<float>(100)
    // gpu::IOAlloc<float>(100)
    float *dx = gpu::alloc<float>(100);
    float *dy = gpu::alloc<float>(100);
    float *dz = gpu::alloc<float>(100);

    gpu::memcpy(dx, hx.data(), 100 * sizeof(float));
    gpu::memcpy(dy, hy.data(), 100 * sizeof(float));

    gpu::runKernel<TestKernels::vectorAdd>(GPUKernelParams{.range={100, 0, 0}}, dx, dy, dz, 100);
    gpu::memcpy(hz.data(), dz, 100 * sizeof(float));

    gpu::free(dx);
    gpu::free(dy);
    gpu::free(dz);

    for (auto &x: hz) {
        if (x != 42) {
            std::cout << "Error: result is " << x << std::endl;
            for (auto &x : hz) {
                std::cout << "Error: result is " << x << std::endl;
            }
            return 1;
        }
    }
    std::cout << "Looking good!" << std::endl;
    return 0;
}