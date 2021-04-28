#ifndef XPU_DRIVER_CUDA_DEVICE_RUNTIME_H
#define XPU_DRIVER_CUDA_DEVICE_RUNTIME_H

#include <cub/cub.cuh>

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

#define XPU_ASSERT(x) static_cast<void>(0)


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

namespace impl {
XPU_D XPU_FORCE_INLINE constexpr float pi() { return 3.14159265358979323846f; }

XPU_D XPU_FORCE_INLINE float ceil(float x) { return ceilf(x); }
XPU_D XPU_FORCE_INLINE float cos(float x) { return cosf(x); }
XPU_D XPU_FORCE_INLINE float fabs(float x) { return fabsf(x); }
XPU_D XPU_FORCE_INLINE float fmin(float a, float b) { return fminf(a, b); }
XPU_D XPU_FORCE_INLINE float fmax(float a, float b) { return fmaxf(a, b); }
XPU_D XPU_FORCE_INLINE int   iabs(int a) { return abs(a); }
XPU_D XPU_FORCE_INLINE int   imin(int a, int b) { return min(a, b); }
XPU_D XPU_FORCE_INLINE int   imax(int a, int b) { return max(a, b); }
XPU_D XPU_FORCE_INLINE float sqrt(float x) { return sqrtf(x); }
XPU_D XPU_FORCE_INLINE float tan(float x) { return tanf(x); }

XPU_D XPU_FORCE_INLINE int atomic_add_block(int *addr, int val) { return atomicAdd(addr, val); }
} // namespace impl

template<typename C>
struct cmem_accessor {
};

template<typename T, size_t N = sizeof(T)>
struct sort_key {};

template<typename T>
struct sort_key<T, 8> {
    using type = unsigned long long int;
    static __device__ constexpr type max() { return 0xFFFFFFFFFFFFFFFFul; }

    static __device__ XPU_FORCE_INLINE bool cmp(type a, type b) {
        return (unsigned int)(a & 0xFFFFFFFF) < (unsigned int)(b & 0xFFFFFFFF);
    }
};

template<typename T>
struct sort_key<T, 4> {
    using type = unsigned int;
    static __device__ constexpr type max() { return 0xFFFFFFFF; }
    static __device__ XPU_FORCE_INLINE bool cmp(type a, type b) {
        return a < b;
    }
};

template<>
struct sort_key<float> {
    using type = float;
    static __device__ constexpr type max() { return INFINITY; }
    static __device__ XPU_FORCE_INLINE bool cmp(type a, type b) {
        return a < b;
    }
};

template<typename T, size_t N>
union local_storage_t {
    T data[N];
    typename sort_key<T>::type as_keys[N];
};

template<typename T, int BlockSize, int ItemsPerThread>
class block_sort_impl {

public:
    using key_t = typename sort_key<T>::type;

    using block_radix_sort = cub::BlockRadixSort<key_t, BlockSize, ItemsPerThread>;

    using storage_t = typename block_radix_sort::TempStorage;

    static_assert(ItemsPerThread == 1, "Can only sort with one item per thread at the moment...");
    static_assert(sizeof(key_t) == sizeof(T));

    __device__ block_sort_impl(storage_t &storage_) : storage(storage_) {}

    __device__ T *sort(T *data, size_t N, T *buf) {
        // if (xpu::thread_idx::x() == 0) {
        //     selection_sort(data, N);
        // }
        return radix_sort(data, N, buf);
    }

private:
    storage_t &storage;

    __device__ T *radix_sort(T *data_, size_t N, T *buf_) {
        const int ItemsPerBlock = BlockSize * ItemsPerThread;

        size_t nItemBlocks = N / ItemsPerBlock + 1;

        key_t *data = reinterpret_cast<key_t *>(data_);
        key_t *buf = reinterpret_cast<key_t *>(buf_);

        key_t tmp_local[ItemsPerThread];

        for (size_t i = 0; i < nItemBlocks; i++) {
            size_t start = i * ItemsPerBlock;
            for (size_t b = 0; b < ItemsPerThread; b++) {
                size_t idx = start + b * ItemsPerThread + thread_idx::x();
                if (idx < N) {
                    tmp_local[b] = data[idx];
                } else {
                    tmp_local[b] = sort_key<T>::max();
                }
            }

            // if (thread_idx::x() == 0 && sizeof(T) == 8) {
            //     printf("BEFORE SORTING\n");
            //     for (size_t k = start; k < start+10; k++) {
            //         printf("%d\n", int(data[k] & 0xFFFFFFFFul));
            //     }
            //     printf("\n");
            // }

            // if (thread_idx::x() == 0) printf("\nBEFORE SORTING\n");
            // for (int t = 0; t < block_dim::x(); t++) {
            //     for (int l = 0; l < ItemsPerThread && t == thread_idx::x(); l++) {
            //         printf("%d, %d: %d\n", t, l, int(tmp_local[l] & 0xFFFFFFFFul));
            //     }
            //     __syncthreads();
            // }

            __syncthreads();
            // FIXME: this call fails with ItemsPerThread > 1???
            block_radix_sort(storage).Sort(tmp_local, 0, 32);
            __syncthreads();

            assert(tmp_local != 0);

            // if (thread_idx::x() == 0) printf("\nAFTER SORTING\n");
            // for (int t = 0; t < block_dim::x(); t++) {
            //     for (int l = 0; l < ItemsPerThread && t == thread_idx::x(); l++) {
            //         printf("%d, %d: %d\n", t, l, int(tmp_local[l] & 0xFFFFFFFFul));
            //     }
            //     __syncthreads();
            // }

            // TODO use block store here
            for (size_t b = 0; b < ItemsPerThread; b++) {
                size_t idx = start + thread_idx::x() * ItemsPerThread + b;
                if (idx < N) {
                    data[idx] = tmp_local[b];
                }
            }

            // __syncthreads();
            // if (thread_idx::x() == 0 && sizeof(T) == 8) {
            //     printf("AFTER SORTING\n");
            //     for (size_t k = start; k < start+10; k++) {
            //         printf("%d\n", int(data[k] & 0xFFFFFFFFul));
            //     }
            //     printf("\n");
            // }
        }

        __syncthreads();

        key_t *src = data;
        key_t *dst = buf;

        // if (thread_idx::x() == 0) {
        //     for (size_t i = N-32; i < N; i++) {
        //         printf("%llu: %llu\n", i, src[i]);
        //     }
        // }

        for (size_t blockSize = ItemsPerBlock; blockSize < N; blockSize *= 2) {

            // if (thread_idx::x() == 0) printf("BEFORE MERGING: ItemsPerBlock = %d,  blockSize = %llu\n", ItemsPerBlock, blockSize);

            size_t carryStart = 0;
            for (size_t st = 0; st + blockSize < N; st += 2 * blockSize) {
                size_t st2 = st + blockSize;
                size_t blockSize2 = min(N - st2, blockSize);
                carryStart = st2 + blockSize2;

                // if (thread_idx::x() == 0 && blockSize2 != blockSize) {
                //     printf("BEFORE MERGING: ItemsPerBlock = %d,  block_size1 = %llu, st1 = %llu, block_size2 = %llu, st2 = %llu\n",
                //         ItemsPerBlock, blockSize, st, blockSize2, st2);

                //     // for (size_t i = st; i < st+10; i++) {
                //     //     printf("%llu: %u\n", i, (unsigned int)(src[i] & 0xFFFFFFFFul));
                //     // }
                //     // for (size_t i = st2; i < st2+10; i++) {
                //     //     printf("%llu: %u\n", i, (unsigned int)(src[i] & 0xFFFFFFFFul));
                //     // }
                // }

                merge(&src[st], &src[st2], blockSize, blockSize2, &dst[st]);


                // for (int i = 0; i < blockSize + blockSize2; i++) {
                //     if (thread_idx::x() == 0 && dst[st + i] == 0) {
                //         printf("FOUND ZERO AFTER MERGING: i = %d,  block_size1 = %llu, st1 = %llu, block_size2 = %llu, st2 = %llu\n",
                //             i, blockSize, st, blockSize2, st2);
                //         break;
                //     }
                // }
            }

            for (size_t i = carryStart + thread_idx::x(); i < N; i += block_dim::x()) {
                dst[i] = src[i];
            }

            __syncthreads();

            key_t *tmp = src;
            src = dst;
            dst = tmp;
        }

        return reinterpret_cast<T *>(src);
    }

    __device__ void merge(const key_t *block1, const key_t *block2, size_t block_size1, size_t block_size2, key_t *out) {
        if (thread_idx::x() > 0) {
            return;
        }

        size_t i1 = 0;
        size_t i2 = 0;
        size_t i_out = 0;

        while (i1 < block_size1 && i2 < block_size2) {
            if (sort_key<T>::cmp(block1[i1], block2[i2])) {
                out[i_out] = block1[i1];
                i1++;
            } else {
                out[i_out] = block2[i2];
                i2++;
            }
            i_out++;
        }

        // if (block_size1 != block_size2) {
        //     printf("MERGING: block_size1 = %llu, i1 = %llu, block_size2 = %llu, i2 = %llu, i_out = %llu\n",
        //         block_size1, i1, block_size2, i2, i_out);
        // }

        size_t r_i = (i1 < block_size1 ? i1 : i2);
        size_t r_size = (i1 < block_size1 ? block_size1 : block_size2);
        const key_t *r_block = (i1 < block_size1 ? block1 : block2);
        for (; r_i < r_size; r_i++, i_out++) {
            out[i_out] = r_block[r_i];
        }
    }

    /*********************************************** KHUN BEGIN *******************************************************/

    enum MgpuBounds {
        MgpuBoundsLower,
        MgpuBoundsUpper
    };


    template<int NT, int VT, typename It1, typename It2, typename Ti, typename Comp>
    __device__ void DeviceMergeKeysIndices(It1 a_global, It2 b_global, int4 range,
    int tid, Ti* keys_shared, Ti* results, int* indices, Comp comp) {
    
        int a0 = range.x;
        int a1 = range.y;
        int b0 = range.z;
        int b1 = range.w;
        int aCount = a1 - a0;
        int bCount = b1 - b0;
    
        // Load the data into shared memory.
        DeviceLoad2ToShared<NT, VT, VT>(a_global + a0, aCount, b_global + b0,
            bCount, tid, keys_shared);
    
        // Run a merge path to find the start of the serial merge for each thread.
        int diag = VT * tid;
        int mp = MergePath<MgpuBoundsLower>(keys_shared, aCount,
            keys_shared + aCount, bCount, diag, comp);
    
        // Compute the ranges of the sources in shared memory.
        int a0tid = mp;
        int a1tid = aCount;
        int b0tid = aCount + diag - mp;
        int b1tid = aCount + bCount;
    
        // Serial merge into register.
        SerialMerge<VT, true>(keys_shared, a0tid, a1tid, b0tid, b1tid, results,
            indices, comp);
    }




    template<MgpuBounds Bounds, typename It1, typename It2, typename Comp>
    __device__ int MergePath(It1 a, int aCount, It2 b, int bCount, int diag,
                                   Comp comp) {

        typedef typename std::iterator_traits<It1>::value_type Tb;
        int begin = max(0, diag - bCount);
        int end = min(diag, aCount);

        while(begin < end) {
            int mid = (begin + end)>> 1;
            Tb aKey = a[mid];
            Tb bKey = b[diag - 1 - mid];
            bool pred = (MgpuBoundsUpper == Bounds) ?
                        comp(aKey, bKey) :
                        !comp(bKey, aKey);
            if(pred) begin = mid + 1;
            else end = mid;
        }
        return begin;
    }



    template<int NT, int VT0, int VT1, typename Tc>
    __device__ void DeviceLoad2ToShared(const Tc* a_global, int aCount,
                                         const Tc* b_global, int bCount, int tid, Tc* shared, bool sync) {

        Tc reg[VT1];
        DeviceLoad2ToReg<NT, VT0, VT1>(a_global, aCount, b_global, bCount, tid,
                                       reg, false);
        DeviceRegToShared<NT, VT1>(reg, tid, shared, sync);
    }

    template<int NT, int VT0, int VT1, typename InputIt1, typename InputIt2,
            typename Td>
    __device__ void DeviceLoad2ToReg(InputIt1 a_global, int aCount,
                                      InputIt2 b_global, int bCount, int tid, Td* reg, bool sync)  {

        b_global -= aCount;
        int total = aCount + bCount;
        if(total >= NT * VT0) {
            #pragma unroll
            for(int i = 0; i < VT0; ++i) {
                int index = NT * i + tid;
                if(index < aCount) reg[i] = a_global[index];
                else reg[i] = b_global[index];
            }
        } else {
            #pragma unroll
            for(int i = 0; i < VT0; ++i) {
                int index = NT * i + tid;
                if(index < aCount) reg[i] = a_global[index];
                else if(index < total) reg[i] = b_global[index];
            }
        }
            #pragma unroll
        for(int i = VT0; i < VT1; ++i) {
            int index = NT * i + tid;
            if(index < aCount) reg[i] = a_global[index];
            else if(index < total) reg[i] = b_global[index];
        }
    }

    template<int NT, int VT0, int VT1, typename InputIt1, typename InputIt2,
            typename Td>
    __device__ void DeviceLoad2ToShared(InputIt1 a_global, int aCount,
                                         InputIt2 b_global, int bCount, int tid, T* shared, bool sync) {

        Td reg[VT1];
        DeviceLoad2ToReg<NT, VT0, VT1>(a_global, aCount, b_global, bCount, tid,
                                       reg, false);
        DeviceRegToShared<NT, VT1>(reg, tid, shared, sync);
    }


    template<int NT, int VT, typename OutputIt, typename Tf>
    __device__ void DeviceRegToShared(const Tf* reg, int tid,
                                       OutputIt dest, bool sync) {

        typedef typename std::iterator_traits<OutputIt>::value_type T2;
        #pragma unroll
        for(int i = 0; i < VT; ++i)
            dest[NT * i + tid] = (T2)reg[i];

        if(sync) __syncthreads();
    }


    template<int VT, bool RangeCheck, typename Tg, typename Comp>
    __device__ void SerialMerge(const Tg* keys_shared, int aBegin, int aEnd,
                                 int bBegin, int bEnd, Tg* results, int* indices, Comp comp) {

        Tg aKey = keys_shared[aBegin];
        Tg bKey = keys_shared[bBegin];

        #pragma unroll
        for(int i = 0; i < VT; ++i) {
            bool p;
            if(RangeCheck)
                p = (bBegin >= bEnd) || ((aBegin < aEnd) && !comp(bKey, aKey));
            else
                p = !comp(bKey, aKey);

            results[i] = p ? aKey : bKey;
            indices[i] = p ? aBegin : bBegin - !RangeCheck;

            if(p) aKey = keys_shared[++aBegin];
            else bKey = keys_shared[++bBegin];
        }
        __syncthreads();
    }

    template <typename TL>
    __device__ void myMergeImpl(TL *a, TL *b, int size){

        TL target = b[threadIdx.x];
        if(threadIdx.x > size){
            return;
        }

        int left = 0, right = size - 1;
        while(left <= right){
            int mid = left + (right - left) >> 1;
            if(a[mid] == target) return mid;
            else if(target > a[mid]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }

    template <typename TT>
    void myMerge(TT *a, int sizeA, TT *b, int sizeB, TT *ret){
        cudaMallocManaged(a, sizeA * sizeof(TT));
        cudaMallocManaged(b, sizeB * sizeof(TT));

        myMergeImpl <<<1, sizeA>>>(a, b, sizeB);;
        cudaDeviceSynchronize();
    }



    /************************************************ KHUN END ********************************************************/



    __device__ void bitonic_sort(T *data, size_t N) {
                // given an array arr of length n, this code sorts it in place
        // all indices run from 0 to n-1
        // for (k = 2; k <= n; k *= 2) // k is doubled every iteration
        //     for (j = k/2; j > 0; j /= 2) // j is halved at every iteration, with truncation of fractional parts
        //         for (i = 0; i < n; i++)
        //             l = bitwiseXOR (i, j); // in C-like languages this is "i ^ j"
        //             if (l > i)
        //                 if (  (bitwiseAND (i, k) == 0) AND (arr[i] > arr[l])
        //                 OR (bitwiseAND (i, k) != 0) AND (arr[i] < arr[l]) )
        //                     swap the elements arr[i] and arr[l]
        for (size_t k = 2; k <= next_power_of_2(N); k *= 2) {
            for (size_t j = k/2; j > 0; j /= 2) {
                for (size_t i = thread_idx::x(); i < N; i += block_dim::x()) {
                    size_t l = i^j;
                    if (l > i && l < N) {
                        if (((i & k) == 0 && data[i] > data[l])
                                || ((i & k) != 0 && data[i] < data[l])) {
                            T tmp = data[i];
                            data[i] = data[l];
                            data[l] = tmp;
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }

    __device__  inline size_t next_power_of_2(size_t N) {
        return N == 1 ? 1 : 1<<(64-__clzll(N-1));
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
