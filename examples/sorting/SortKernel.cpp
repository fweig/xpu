// Include auto-generated header, where the kernel is declared.
#include "SortKernel.h"

// Optional shorthand for the sorting class.
//
// Template arguments are the type of the key that is sorted,
// size of the gpu block (currently hard-coded at 64 threads)
// and the number of keys that are sorted by each thread with
// the underlying cub::BlockRadixSort implementation.
using SortT = xpu::block_sort<float, 64, 4>;

// Define type that is used to allocate shared memory.
// In this case only shared memory for the underlying cub::BlockRadixSort is needed.
struct GpuSortSmem {
    typename SortT::storage_t sortBuf;
};

// Kernel implementation.
XPU_KERNEL(SortKernel, gpuSort, GpuSortSmem, (KeyValuePair *) data, (KeyValuePair *) buf, (KeyValuePair **) out, (size_t) numElems) {
    // Call the sort function. Along the two buffers and the number of elements, a function that
    // extracts the key from the struct has to be passed.
    // Returns the buffer that contains the sorted data (either data or buf).
    *out = SortT(shm.sortBuf).sort<KeyValuePair>(
        data, numElems, buf,
        [](const KeyValuePair &dat) { return dat.key; }
    );
}
