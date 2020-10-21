#include "VectorOps.h"
#include <xpu/xpu.h>

#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    constexpr int NElems = 100;

    xpu::initialize(xpu::driver::cuda); // or xpu::driver::cuda

    std::vector<float> hx(NElems, 8);
    std::vector<float> hy(NElems, 8);

    float *dx = xpu::device_malloc<float>(NElems);
    float *dy = xpu::device_malloc<float>(NElems);
    float *dz = xpu::device_malloc<float>(NElems);

    xpu::memcpy(dx, hx.data(), NElems * sizeof(float));
    xpu::memcpy(dy, hy.data(), NElems * sizeof(float));

    xpu::run_kernel<VectorOps::add>(xpu::grid::n_threads(NElems), dx, dy, dz, NElems);

    std::vector<float> hz(NElems);

    xpu::memcpy(hz.data(), dz, NElems * sizeof(float));

    for (auto &x: hz) {
        if (x != 16) {
            std::cout << "ERROR";
            abort();
        }
    }

    xpu::free(dx);
    xpu::free(dy);
    xpu::free(dz);

    std::cout << "Looking good!" << std::endl;
    return 0;
}