// Include header where the kernel is declared.
#include "SortKernel.h"

// Initialize the xpu image. This macro must be placed once somewhere in the device sources.
XPU_IMAGE(SortKernel);

// Export kernel.
XPU_EXPORT(GpuSort);
// Kernel implementation.
XPU_D void GpuSort::operator()(context &ctx,  KeyValuePair *data, KeyValuePair *buf, KeyValuePair **out, size_t numElems) {
    // Call the sort function. Along the two buffers and the number of elements, a function that
    // extracts the key from the struct has to be passed.
    // Returns the buffer that contains the sorted data (either data or buf).

    size_t itemsPerBlock = numElems / xpu::grid_dim::x();
    size_t offset = itemsPerBlock * xpu::block_idx::x();

    KeyValuePair *res = SortT(ctx.smem()).sort(
        &data[offset], itemsPerBlock, &buf[offset],
        [](const KeyValuePair &dat) { return dat.key; }
    );

    if (xpu::block_idx::x() == 0) {
        *out = res;
    }
}
