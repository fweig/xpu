#include "MergeKernel.h"

#include <xpu/host.h>

int main() {

    static constexpr size_t N = 1024;

    xpu::initialize();

    xpu::hd_buffer<float> a{N};
    xpu::hd_buffer<float> b{N};
    xpu::hd_buffer<float> dst{a.size() + b.size()};

    for (size_t i = 0; i < N; i++) {
        a.host()[i] = 2*i;
        b.host()[i] = 2*i+1;
    }

    xpu::copy(a, xpu::host_to_device);
    xpu::copy(b, xpu::host_to_device);

    xpu::run_kernel<GpuMerge>(xpu::grid::n_blocks(1), a.device(), a.size(), b.device(), b.size(), dst.device());

    xpu::copy(dst, xpu::device_to_host);

    float *h = dst.host();
    bool isSorted = true;
    for (size_t i = 1; i < dst.size(); i++) {
        isSorted &= (h[i-1] <= h[i]);
    }

    if (isSorted) {
        std::cout << "Data is sorted!" << std::endl;
    } else {
        for (size_t i = 0; i < dst.size(); i++) {
            std::cout << h[i] << " ";
            if (i % 10 == 9) {
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
        std::cout << "ERROR: Data is not sorted!" << std::endl;
    }

    return 0;
}
