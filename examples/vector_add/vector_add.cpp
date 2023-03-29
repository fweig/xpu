#include "VectorOps.h"

#include <xpu/host.h>

#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    constexpr int NElems = 100;

    xpu::initialize();

    xpu::device_prop prop{xpu::device::active()};
    std::cout << "Running VectorAdd on Device '" << prop.name() << "' " <<  std::endl;

    constexpr int NElems = 1000;
    xpu::buffer<float> x{NElems, xpu::io_buffer};
    xpu::buffer<float> y{NElems, xpu::io_buffer};
    xpu::buffer<float> z{NElems, xpu::io_buffer};

    xpu::h_view xh{x};
    xpu::h_view yh{y};

    for (int i = 0; i < NElems; i++) {
        xh[i] = i;
        yh[i] = i;
    }

    xpu::copy(x, xpu::host_to_device);
    xpu::copy(y, xpu::host_to_device);

    xpu::run_kernel<VectorAdd>(xpu::n_threads(NElems), x.d(), y.d(), z.d(), NElems);

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
