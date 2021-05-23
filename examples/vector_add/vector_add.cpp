#include "VectorOps.h"

#include <xpu/host.h>

#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    constexpr int NElems = 100;

    xpu::initialize(xpu::driver::cuda); // or xpu::driver::cuda

    xpu::hd_buffer<float> x{NElems};
    std::fill_n(x.host(), NElems, 8);
    xpu::hd_buffer<float> y{NElems};
    std::fill_n(y.host(), NElems, 8);
    xpu::hd_buffer<float> z{NElems};

    xpu::copy(x, xpu::host_to_device);
    xpu::copy(y, xpu::host_to_device);

    xpu::run_kernel<VectorAdd>(xpu::grid::n_threads(NElems), x.device(), y.device(), z.device(), NElems);

    xpu::copy(z, xpu::device_to_host);

    for (int i = 0; i < NElems; i++) {
        if (z.host()[i] != 16) {
            std::cout << "ERROR" << std::endl;
            abort();
        }
    }

    std::cout << "Looking good!" << std::endl;
    return 0;
}
