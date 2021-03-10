#ifndef XPU_DRIVER_CUDA_DEVICE_RUNTIME_H
#define XPU_DRIVER_CUDA_DEVICE_RUNTIME_H

#ifndef XPU_DEVICE_H
#error "This header should not be included directly. Include xpu/device.h instead."
#endif

#include "../../detail/macros.h"


#define XPU_CMEM_IDENTIFIER(name) XPU_CONCAT(xpu_cuda_driver_cmem_symbol_, name)

#if XPU_IS_CUDA
#define XPU_CHOOSE(hip, cuda) cuda
#else
#define XPU_CHOOSE(hip, cuda) hip
#endif

#if XPU_IS_CUDA
#define XPU_INTERNAL_LAUNCH_KERNEL(name, nBlocks, nThreads, ...) \
    name<<<(nBlocks), (nThreads)>>>(XPU_PARAM_NAMES(() 0, ##__VA_ARGS__))
#else
#define XPU_INTERNAL_LAUNCH_KERNEL(name, nBlocks, nThreads, ...) \
    hipLaunchKernelGGL(name, dim3(nBlocks), dim3(nThreads), 0, 0, XPU_PARAM_NAMES(() 0, ##__VA_ARGS__))
#endif

#if XPU_IS_CUDA
#define XPU_INTERNAL_SUFFIX _Cuda
#else
#define XPU_INTERNAL_SUFFIX _Hip
#endif

// TODO: don't hardcode block size
#define XPU_DETAIL_KERNEL(deviceLibrary, name, sharedMemoryT, ...) \
    __device__ void name ## _impl(XPU_PARAM_LIST((const xpu::kernel_info &) info, (sharedMemoryT &) smem, ##__VA_ARGS__)); \
    __global__ void name ## _entry(XPU_PARAM_LIST((char) /*dummyArg*/, ##__VA_ARGS__)) { \
        __shared__ sharedMemoryT shm; \
        xpu::kernel_info info{ \
            .i_thread = xpu::dim{xpu::thread_idx::x(), 0, 0}, \
            .n_threads  = xpu::dim{xpu::block_dim::x(), 0, 0}, \
            .i_block  = xpu::dim{xpu::block_idx::x(), 0, 0}, \
            .n_blocks   = xpu::dim{xpu::grid_dim::x(), 0, 0} \
        }; \
        name ## _impl(XPU_PARAM_NAMES(() info, () shm, ##__VA_ARGS__)); \
    } \
    xpu::detail::error XPU_CONCAT(deviceLibrary, XPU_INTERNAL_SUFFIX)::run_ ## name(XPU_PARAM_LIST((xpu::grid) params, ##__VA_ARGS__)) { \
        printf("Running Kernel " #name "\n"); \
        if (params.threads.x > -1) { \
            XPU_INTERNAL_LAUNCH_KERNEL(name ## _entry, (params.threads.x + 63) / 64, 64, ##__VA_ARGS__); \
        } else { \
            XPU_INTERNAL_LAUNCH_KERNEL(name ## _entry, params.blocks.x, 64, ##__VA_ARGS__); \
        } \
        auto err = XPU_CHOOSE(hipDeviceSynchronize(), cudaDeviceSynchronize()); \
        if (err != 0) { \
            printf("Kernel Error: %s\n", XPU_CHOOSE(hipGetErrorString(err), cudaGetErrorString(err))); \
        } \
        return err; \
    } \
    __device__ inline void name ## _impl( XPU_PARAM_LIST((const xpu::kernel_info &) info, (sharedMemoryT &) shm, ##__VA_ARGS__))

#define XPU_DETAIL_ASSERT(x) static_cast<void>(0)


namespace xpu {

XPU_D XPU_FORCE_INLINE int thread_idx::x() {
    return XPU_CHOOSE(hipThreadIdx_x, threadIdx.x);
}

XPU_D XPU_FORCE_INLINE int block_dim::x() {
    return XPU_CHOOSE(hipBlockDim_x, blockDim.x);
}

XPU_D XPU_FORCE_INLINE int block_idx::x() {
    return XPU_CHOOSE(hipBlockIdx_x, blockIdx.x);
}

XPU_D XPU_FORCE_INLINE int grid_dim::x() {
    return XPU_CHOOSE(hipGridDim_x, gridDim.x);
}

// XPU_D XPU_FORCE_INLINE constexpr float pi() { return 3.14159265358979323846f; }

XPU_D XPU_FORCE_INLINE float ceil(float x) { return ::ceilf(x); }
XPU_D XPU_FORCE_INLINE float cos(float x) { return ::cosf(x); }
XPU_D XPU_FORCE_INLINE float abs(float x) { return ::fabsf(x); }
XPU_D XPU_FORCE_INLINE float min(float a, float b) { return ::fminf(a, b); }
XPU_D XPU_FORCE_INLINE float max(float a, float b) { return ::fmaxf(a, b); }
XPU_D XPU_FORCE_INLINE int   abs(int a) { return ::abs(a); }
XPU_D XPU_FORCE_INLINE int   min(int a, int b) { return ::min(a, b); }
XPU_D XPU_FORCE_INLINE unsigned long long int min(unsigned long long int a, unsigned long long int b) { return ::min(a, b); }
XPU_D XPU_FORCE_INLINE long long int min(long long int a, long long int b) { return ::min(a, b); }
XPU_D XPU_FORCE_INLINE int   max(int a, int b) { return ::max(a, b); }
XPU_D XPU_FORCE_INLINE float sqrt(float x) { return ::sqrtf(x); }
XPU_D XPU_FORCE_INLINE float tan(float x) { return ::tanf(x); }

XPU_D XPU_FORCE_INLINE int atomic_add_block(int *addr, int val) { return atomicAdd(addr, val); }

template<typename C>
struct cmem_accessor {
};

template<typename C> XPU_D XPU_FORCE_INLINE const C &cmem() { return cmem_accessor<C>::get(); }

} // namespace xpu

#endif
