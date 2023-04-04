#include "VectorOps.h"

#include <xpu/host.h>

#include <cassert>
#include <algorithm>
#include <iostream>
#include <vector>

int main() {

    xpu::initialize();

    xpu::device_prop prop{xpu::device::active()};
    std::cout << "Running VectorAdd on Device '" << prop.name() << "' " <<  std::endl;

    constexpr int NElems = 1000;
    xpu::buffer<float> x{NElems, xpu::buf_io};
    xpu::buffer<float> y{NElems, xpu::buf_io};
    xpu::buffer<float> z{NElems, xpu::buf_io};

    xpu::h_view xh{x};
    xpu::h_view yh{y};

    for (int i = 0; i < NElems; i++) {
        xh[i] = i;
        yh[i] = i;
    }


    xpu::queue q{};

    q.copy(x, xpu::host_to_device);
    q.copy(y, xpu::host_to_device);
    q.launch<VectorAdd>(xpu::n_threads(NElems), x, y, z, NElems);
    q.copy(z, xpu::device_to_host);
    q.wait();


    xpu::h_view zh{z};
    for (int i = 0; i < NElems; i++) {
        if (zh[i] != 2 * i) {
            std::cout << "ERROR " << i << " " << zh[i] << std::endl;
            abort();
        }
    }

    std::cout << "Looking good!" << std::endl;
    return 0;
}
