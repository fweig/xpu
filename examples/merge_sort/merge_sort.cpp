
 #include <iostream>
 #include "mergeSortOps.h"
 #include <xpu/host.h>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <vector>

int main() {
    std::cout << "0" << std::endl;
    constexpr int NElems = 1000;

    std::mt19937 gen{1337};
    std::uniform_int_distribution<unsigned int> dist{0, 10000};
    std::cout << "1" << std::endl;
    std::vector<unsigned int> items{};
    for (size_t i = 0; i < NElems; i++) {
        items.emplace_back(dist(gen));
    }

    float *dx = xpu::device_malloc<float>(NElems);
    free(dx);
    // std::cout << "2" << std::endl;
    // unsigned int *ditems = xpu::device_malloc<unsigned int>(NElems);
    // std::cout << "2.1" << std::endl;
    // unsigned int *buf = xpu::device_malloc<unsigned int>(NElems);
    // std::cout << "2.2" << std::endl;
    // unsigned int **dst = xpu::device_malloc<unsigned int *>(1);
    // std::cout << "3" << std::endl;
    // xpu::copy(ditems, items.data(), NElems);



    // xpu::run_kernel<mergeSortOps::sort>(xpu::grid::n_blocks(1), ditems, NElems, buf, dst);

    // unsigned int *hdst = nullptr;
    // xpu::copy(&hdst, dst, 1);
    // xpu::copy(items.data(), hdst, NElems);

    //block_sort_impl srt;
    
    return 0;
}