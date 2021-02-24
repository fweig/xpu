#ifndef XPU_DRIVER_CUDA_DEVICE_RUNTIME_H
#define XPU_DRIVER_CUDA_DEVICE_RUNTIME_H

#define XPU_CMEM_IDENTIFIER(name) XPU_CONCAT(xpu_cuda_driver_cmem_symbol_, name) 

// TODO: don't hardcode block size
#define XPU_KERNEL(deviceLibrary, name, sharedMemoryT, ...) \
    __device__ void name ## _impl(XPU_PARAM_LIST((const xpu::kernel_info &) info, (sharedMemoryT &) smem, ##__VA_ARGS__)); \
    __global__ void name ## _entry(XPU_PARAM_LIST((char) /*dummyArg*/, ##__VA_ARGS__)) { \
        __shared__ sharedMemoryT shm; \
        xpu::kernel_info info{ \
            .i_thread = xpu::dim{int(threadIdx.x), 0, 0}, \
            .n_threads  = xpu::dim{int(blockDim.x), 0, 0}, \
            .i_block  = xpu::dim{int(blockIdx.x), 0, 0}, \
            .n_blocks   = xpu::dim{int(gridDim.x), 0, 0} \
        }; \
        name ## _impl(XPU_PARAM_NAMES(() info, () shm, ##__VA_ARGS__)); \
    } \
    xpu::error deviceLibrary##_Cuda::run_ ## name(XPU_PARAM_LIST((xpu::grid) params, ##__VA_ARGS__)) { \
        if (params.threads.x > -1) { \
            name ## _entry<<<(params.threads.x + 31) / 32, 32>>>(XPU_PARAM_NAMES(() 0, ##__VA_ARGS__)); \
        } else { \
            name ## _entry<<<params.blocks.x, 32>>>(XPU_PARAM_NAMES(() 0, ##__VA_ARGS__)); \
        } \
        cudaDeviceSynchronize(); \
        return 0; \
    } \
    __device__ inline void name ## _impl( XPU_PARAM_LIST((const xpu::kernel_info &) info, (sharedMemoryT &) shm, ##__VA_ARGS__))

#define XPU_ASSERT(x) static_cast<void>(0)

namespace xpu {

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
        if (threadIdx.x == 0) {
            selection_sort(data, N);
        }
    }

private:
    __device__ void selection_sort(T *data, size_t N) {
        for (int i = 0 ; i < N ; ++i) {
            T min_val = data[i];
            size_t min_idx = i;

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