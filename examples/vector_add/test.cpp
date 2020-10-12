#include "VectorOps.h"
#include <xpu/xpu.h>

#include <algorithm>
#include <iostream>
#include <vector>

int main() {

    xpu::initialize(xpu::driver::cuda); // or xpu::driver::cuda or xpu::driver::hip or xpu::driver::sycl

    std::vector<float> hx(100);
    std::fill(hx.begin(), hx.end(), 16);
    std::vector<float> hy(100);
    std::fill(hy.begin(), hy.end(), 26);
    std::vector<float> hz(100);

    // xpu::h_ptr<float> hx = xpu::ptr_cast<xpu::h_ptr<float>>(x);
    // xpu::h_ptr<float> hx = xpu::host_malloc<float>(100)
    // xpu::hd_ptr<float> hdx = xpu::host_device_malloc(100)
    // xpu::d_ptr<float> dx = xpu::device_malloc<float>(100)
    // xpu::hd_ptr<float> hdx = xpu::glue(hx, dx, mem_id)
    // xpu::pinned_ptr<float>
    float *dx = xpu::device_malloc<float>(100);
    float *dy = xpu::device_malloc<float>(100);
    float *dz = xpu::device_malloc<float>(100);

    xpu::memcpy(dx, hx.data(), 100 * sizeof(float));
    xpu::memcpy(dy, hy.data(), 100 * sizeof(float));

    xpu::run_kernel<VectorOps::vectorAdd>(xpu::grid::n_threads(100), dx, dy, dz, 100);
    xpu::memcpy(hz.data(), dz, 100 * sizeof(float));

    xpu::free(dx);
    xpu::free(dy);
    xpu::free(dz);

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