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

    xpu::tpos &pos = ctx.pos();

    size_t itemsPerBlock = numElems / pos.grid_dim_x();
    size_t offset = itemsPerBlock * pos.block_idx_x();

    KeyValuePair *res = SortT(pos, ctx.smem()).sort(
        &data[offset], itemsPerBlock, &buf[offset],
        [](const KeyValuePair &dat) { return dat.key; }
    );

    if (pos.block_idx_x() == 0) {
        *out = res;
    }
}
