#ifndef SORT_KERNEL_H
#define SORT_KERNEL_H

#include "KeyValuePair.h"
#include <xpu/device.h>
#include <cstddef> // for size_t

struct SortKernel {};


// Define GpuSort-Kernel
struct GpuSort : xpu::kernel<SortKernel> {

    // Optional shorthand for the sorting class.
    //
    // Template arguments are the type of the key that is sorted,
    // size of the gpu block (currently hard-coded at 64 threads)
    // and the number of keys that are sorted by each thread with
    // the underlying cub::BlockRadixSort implementation.
    using SortT = xpu::block_sort<float, KeyValuePair, 64, 4>;

    // Define shared memory
    using shared_memory = SortT::storage_t;

    using context = xpu::kernel_context<shared_memory>;
    XPU_D void operator()(context &, KeyValuePair *, KeyValuePair *, KeyValuePair **, size_t size);
};

#endif
