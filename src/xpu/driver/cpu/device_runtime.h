#ifndef XPU_DRIVER_CPU_DEVICE_RUNTIME
#define XPU_DRIVER_CPU_DEVICE_RUNTIME

#include <algorithm>

#define XPU_KERNEL(name, sharedMemoryT, ...) \
    void kernel_ ## name(XPU_PARAM_LIST((const xpu::kernel_info &) info, (sharedMemoryT &) smem, ##__VA_ARGS__)); \
    xpu::error XPU_CONCAT(XPU_DEVICE_LIBRARY, XPU_DRIVER_NAME)::run_ ## name(XPU_PARAM_LIST((xpu::grid) params, ##__VA_ARGS__)) { \
        if (params.threads.x == -1) { \
            params.threads.x = params.blocks.x; \
        } \
        for (int i = 0; i < params.threads.x; i++) { \
            sharedMemoryT shm{}; \
            xpu::kernel_info info{ \
            .i_thread = {0, 0, 0}, \
            .n_threads = {1, 0, 0}, \
            .i_block = {i, 0, 0}, \
            .n_blocks = {params.threads.x, 0, 0}}; \
            \
            kernel_ ## name(XPU_PARAM_NAMES(() info, () shm, ##__VA_ARGS__)); \
        } \
        return 0; \
    } \
    \
    void kernel_ ## name(XPU_PARAM_LIST((__attribute__((unused)) const xpu::kernel_info &) info, (__attribute__((unused)) sharedMemoryT &) shm, ##__VA_ARGS__))

namespace xpu {

template<typename T, int BlockSize>
class block_sort_impl {

public:
    struct storage {};

    block_sort_impl(storage &) {}

    void sort(T *vals, size_t N) {
        std::sort(vals, &vals[N]);
    }

};

} // namespace xpu

#endif