#ifndef XPU_BENCH_DEVICE_H
#define XPU_BENCH_DEVICE_H

#include <xpu/device.h>

struct bench_device : xpu::device_image {};

template<int elems_per_thread>
struct merge : xpu::kernel<bench_device> {
    using block_size    = xpu::block_size<
        #if XPU_IS_CUDA
            32
        #else
            64
        #endif
    >;

    using merge_t       = xpu::block_merge<float, block_size::value.x, elems_per_thread>;
    using shared_memory = typename merge_t::storage_t;
    using context       = xpu::kernel_context<shared_memory>;
    XPU_D void operator()(context &, const float *a, const float *b, size_t N, float *c);
};

template<int elems_per_thread>
struct sort : xpu::kernel<bench_device> {
    using block_size    = xpu::block_size<
        #if XPU_IS_CUDA
            32
        #else
            64
        #endif
    >;

    using sort_t        = xpu::block_sort<float, float, block_size::value.x, elems_per_thread>;
    using shared_memory = typename sort_t::storage_t;
    using context       = xpu::kernel_context<shared_memory>;
    XPU_D void operator()(context &, float *a, size_t N, float *b, float **c);
};

#endif
