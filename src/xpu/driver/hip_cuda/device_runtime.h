#ifndef XPU_DRIVER_CUDA_DEVICE_RUNTIME_H
#define XPU_DRIVER_CUDA_DEVICE_RUNTIME_H

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
#define XPU_KERNEL(deviceLibrary, name, sharedMemoryT, ...) \
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
    xpu::error XPU_CONCAT(deviceLibrary, XPU_INTERNAL_SUFFIX)::run_ ## name(XPU_PARAM_LIST((xpu::grid) params, ##__VA_ARGS__)) { \
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

#define XPU_ASSERT(x) static_cast<void>(0)


namespace xpu {

XPU_D XPU_INLINE int thread_idx::x() { 
    return XPU_CHOOSE(hipThreadIdx_x, threadIdx.x); 
}

XPU_D XPU_INLINE int block_dim::x() {
    return XPU_CHOOSE(hipBlockDim_x, blockDim.x);
}

XPU_D XPU_INLINE int block_idx::x() {
    return XPU_CHOOSE(hipBlockIdx_x, blockIdx.x);
}

XPU_D XPU_INLINE int grid_dim::x() {
    return XPU_CHOOSE(hipGridDim_x, gridDim.x);
}

namespace impl {
__device__ __forceinline__ constexpr float pi() { return 3.14159265358979323846f; }

__device__ __forceinline__ float ceil(float x) { return ceilf(x); }
__device__ __forceinline__ float cos(float x) { return cosf(x); }
__device__ __forceinline__ float fabs(float x) { return fabsf(x); }
__device__ __forceinline__ float fmin(float a, float b) { return fminf(a, b); }
__device__ __forceinline__ float fmax(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ int   iabs(int a) { return abs(a); }
__device__ __forceinline__ int   imin(int a, int b) { return min(a, b); }
__device__ __forceinline__ int   imax(int a, int b) { return max(a, b); }
__device__ __forceinline__ float sqrt(float x) { return sqrtf(x); }
__device__ __forceinline__ float tan(float x) { return tanf(x); }
} // namespace impl

template<typename C>
struct cmem_accessor {
};

template<typename T, size_t BlockSize>
class block_sort_impl {

public:
    struct storage {};

    __device__ block_sort_impl(storage &) {}

    __device__ void sort(T *data, size_t N) {
        if (xpu::thread_idx::x() == 0) {
            // min(data, N);
            selection_sort(data, N);
        }
    }

private:
    __device__ void min(T *data, size_t N) {
        for (int i = 1; i < N; i++) {
            if (data[i] < data[0]) {
                data[0] = data[i];
            }
        }
    }

    __device__ void selection_sort(T *data, size_t N) {
        for (int i = 0 ; i < N ; ++i) {
            T min_val = data[i];
            int min_idx = i;

            // Find the smallest value in the range [left, right].
            for (int j = i+1 ; j < N ; ++j) {
                T val_j = data[j];

                if (val_j < min_val) {
                    min_idx = j;
                    min_val = val_j;
                }
            }

            // Swap the values.
            if (i != min_idx) {
                data[min_idx] = data[i];
                data[i] = min_val;
            }
        }
    }

};

} // namespace xpu

#endif