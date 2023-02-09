#include "VectorOps.h"

#include <xpu/host.h>

#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    constexpr int NElems = 100;

    xpu::initialize();

    xpu::hd_buffer<float> x{NElems};
    std::fill_n(x.h(), NElems, 8);
    xpu::hd_buffer<float> y{NElems};
    std::fill_n(y.h(), NElems, 8);
    xpu::hd_buffer<float> z{NElems};

    xpu::copy(x, xpu::host_to_device);
    xpu::copy(y, xpu::host_to_device);

    xpu::run_kernel<VectorAdd>(xpu::grid::n_threads(NElems), x.d(), y.d(), z.d(), NElems);

    xpu::copy(z, xpu::device_to_host);

    for (int i = 0; i < NElems; i++) {
        if (z.h()[i] != 16) {
            std::cout << "ERROR " << i << " " << z.h()[i] << " " << z.d()[i] << std::endl;
            abort();
        }
    }

    std::cout << "Looking good!" << std::endl;
    return 0;
}
