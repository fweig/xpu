#ifndef XPU_DRIVER_CUDA_DEVICE_RUNTIME_H
#define XPU_DRIVER_CUDA_DEVICE_RUNTIME_H

#ifndef XPU_DEVICE_LIBRARY_BACKEND_NAME
#error "Backend name missing."
#endif

// TODO: don't hardcode block size
#define XPU_KERNEL(name, sharedMemoryT, ...) \
    __device__ void name ## _impl(const xpu::kernel_info &info, sharedMemoryT &, XPU_PARAM_LIST(__VA_ARGS__)); \
    __global__ void name ## _entry(XPU_PARAM_LIST(__VA_ARGS__)) { \
        __shared__ sharedMemoryT shm; \
        xpu::kernel_info info{ \
            .i_thread = xpu::dim{int(threadIdx.x), 0, 0}, \
            .n_threads  = xpu::dim{int(blockDim.x), 0, 0}, \
            .i_block  = xpu::dim{int(blockIdx.x), 0, 0}, \
            .n_blocks   = xpu::dim{int(gridDim.x), 0, 0} \
        }; \
        name ## _impl(info, shm, XPU_PARAM_NAMES(__VA_ARGS__)); \
    } \
    xpu::error XPU_DEVICE_LIBRARY_BACKEND_NAME::run_ ## name(xpu::grid params, XPU_PARAM_LIST(__VA_ARGS__)) { \
        name ## _entry<<<(params.threads.x + 31) / 32, 32>>>(XPU_PARAM_NAMES(__VA_ARGS__)); \
        return 0; \
    } \
    __device__ inline void name ## _impl(const xpu::kernel_info &info, sharedMemoryT &shm, XPU_PARAM_LIST(__VA_ARGS__))

namespace xpu {

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