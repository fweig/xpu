#include "MergeKernel.h"

#include <xpu/host.h>

int main() {

    static constexpr size_t N = 1024;

    xpu::initialize();

    xpu::buffer<float> a{N, xpu::io_buffer};
    xpu::buffer<float> b{N, xpu::io_buffer};
    xpu::buffer<float> dst{N + N, xpu::io_buffer};

    xpu::h_view a_h{a};
    xpu::h_view b_h{b};
    for (size_t i = 0; i < N; i++) {
        a_h[i] = 2*i;
        b_h[i] = 2*i+1;
    }

    xpu::copy(a, xpu::host_to_device);
    xpu::copy(b, xpu::host_to_device);

    xpu::run_kernel<GpuMerge>(xpu::n_blocks(1), a.get(), N, b.get(), N, dst.get());

    xpu::copy(dst, xpu::device_to_host);

    xpu::h_view h{dst};
    bool isSorted = true;
    for (size_t i = 1; i < h.size(); i++) {
        isSorted &= (h[i-1] <= h[i]);
    }

    if (isSorted) {
        std::cout << "Data is sorted!" << std::endl;
    } else {
        for (size_t i = 0; i < h.size(); i++) {
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
