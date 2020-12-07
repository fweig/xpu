#ifndef XPU_DRIVER_CUDA_DEVICE_RUNTIME_H
#define XPU_DRIVER_CUDA_DEVICE_RUNTIME_H

#ifndef XPU_DEVICE_LIBRARY_BACKEND_NAME
#error "Backend name missing."
#endif

// TODO: don't hardcode block size
#define XPU_KERNEL(name, sharedMemoryT, ...) \
    __device__ void name ## _impl(const xpu::kernel_info &info, sharedMemoryT &, PARAM_LIST(__VA_ARGS__)); \
    __global__ void name ## _entry(PARAM_LIST(__VA_ARGS__)) { \
        __shared__ sharedMemoryT shm; \
        xpu::kernel_info info{ \
            .i_thread = xpu::dim{int(threadIdx.x), 0, 0}, \
            .n_threads  = xpu::dim{int(blockDim.x), 0, 0}, \
            .i_block  = xpu::dim{int(blockIdx.x), 0, 0}, \
            .n_blocks   = xpu::dim{int(gridDim.x), 0, 0} \
        }; \
        name ## _impl(info, shm, PARAM_NAMES(__VA_ARGS__)); \
    } \
    xpu::error XPU_DEVICE_LIBRARY_BACKEND_NAME::run_ ## name(xpu::grid params, PARAM_LIST(__VA_ARGS__)) { \
        name ## _entry<<<(params.threads.x + 31) / 32, 32>>>(PARAM_NAMES(__VA_ARGS__)); \
        return 0; \
    } \
    __device__ inline void name ## _impl(const xpu::kernel_info &info, sharedMemoryT &shm, PARAM_LIST(__VA_ARGS__))

#endif