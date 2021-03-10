#ifndef XPU_BLOCK_OPS_H

// FIXME this header should just be part of device.h,
// but thats currently not possible because it would cause circular dependencies.

#include "device.h"

#if XPU_IS_HIP_CUDA
#include "driver/hip_cuda/block_ops.h"
#else // CPU
#include "driver/cpu/block_ops.h"
#endif

namespace xpu {

template<typename T, int BlockSize, int ItemsPerThread=8>
class block_sort {

    static_assert(sizeof(T) <= 8, "block_sort can only sort keys with up to 8 bytes...");

public:
    using impl_t = block_sort_impl<T, BlockSize, ItemsPerThread>;
    using storage_t = typename impl_t::storage_t;

    XPU_D XPU_FORCE_INLINE block_sort(storage_t &storage) : impl(storage) {}

    XPU_D XPU_FORCE_INLINE T *sort(T *vals, size_t N, T *buf) { return impl.sort(vals, N, buf); }

private:
    impl_t impl;

};

} // namespace xpu

#endif
