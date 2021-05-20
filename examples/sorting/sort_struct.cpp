// Include auto-generated header, where the kernel is declared.
#include "SortKernel.h"
#include "KeyValuePair.h"

// Include host functions to control the GPU.
#include <xpu/host.h>

// STL includes
#include <iostream>
#include <random>
#include <vector>

int main() {

    // Number of elements to sort.
    constexpr size_t NumElems = 1000;

    // Initialize the xpu runtime and select cuda backend.
    xpu::initialize(xpu::driver::cuda);

    // Host buffer.
    std::vector<KeyValuePair> itemsH;

    // Fill host buffer with random numbers.
    std::mt19937 gen{42};
    std::uniform_real_distribution<float> dist{0, 1000000};

    for (size_t i = 0; i < NumElems; i++) {
        itemsH.push_back(KeyValuePair{dist(gen), dist(gen)});
    }

    // Allocate memory on gpu for sorting.
    // Due to the merge step, sorting is not guaranteed to happen in-place.
    // Thus three buffers are required:
    // - inputD: contains the input data.
    // - bufD:   additional buffer that is used by the sorting algorithm,
    //           must have the same size as inputD.
    // - outD: buffer that contains a single pointer which points to the sorted data.
    //         (This will be either inputD or bufD.)
    KeyValuePair *inputD = xpu::device_malloc<KeyValuePair>(NumElems);
    KeyValuePair *bufD   = xpu::device_malloc<KeyValuePair>(NumElems);
    KeyValuePair **outD  = xpu::device_malloc<KeyValuePair *>(1);

    // Copy data from host to GPU.
    xpu::copy(inputD, itemsH.data(), NumElems);

    // Run kernel that performs the sorting.
    //xpu::run_kernel<SortKernel::gpuSort>(xpu::grid::n_blocks(1), inputD, bufD, outD, indices, shared_keys, NumElems);
    //KHUN BEGIN
    xpu::run_kernel<SortKernel::gpuSort>(xpu::grid::n_blocks(1), inputD, bufD, outD, NumElems);
    //KHUN END

    // Get the buffer that contains the sorted data.
    KeyValuePair *outH = nullptr;
    xpu::copy(&outH, outD, 1);

    // Copy sorted data back to host.
    xpu::copy(itemsH.data(), outH, NumElems);

    // Check if data is sorted.
    bool ok = true;
    for (size_t i = 1; i < itemsH.size(); i++) {
        auto faa =  (itemsH[i-1].key <= itemsH[i].key);
        ok &= faa;
        if(faa == false){
            std::cout << i << std::endl;
        }
    }

    if (ok) {
        std::cout << "Data is sorted!" << std::endl;
    } else {
        std::cout << "Error: Data is not sorted!" << std::endl;
    }

    // for(int i = 0; i<((int)NumElems/10); i++){
    //     for(int j = 0; j<10; j++){
    //         std::cout << itemsH[i*1+j].key << ", ";
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << "Cleaning" << std::endl;

    // Cleanup: Free data allocated on GPU.
    xpu::free(inputD);
    xpu::free(bufD);
    xpu::free(outD);




    return 0;
}
