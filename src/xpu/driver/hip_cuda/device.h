
#ifndef XPU_DRIVER_CUDA_DEVICE_RUNTIME_H
#define XPU_DRIVER_CUDA_DEVICE_RUNTIME_H



#ifndef XPU_DEVICE_H
#error "This header should not be included directly. Include xpu/device.h instead."
#endif

#include "../../detail/macros.h"

#if XPU_IS_CUDA
#include <cub/cub.cuh>
#else // XPU_IS_HIP
#include <hipcub/hipcub.hpp>
#endif

#include <iostream>

#if 0
#define PRINT_B(message, ...) if (xpu::thread_idx::x() == 0) printf("t 0: " message "\n", ##__VA_ARGS__)
#define PRINT_T(message, ...) do { printf("t %d: " message "\n", xpu::thread_idx::x(), ##__VA_ARGS__); } while (0)
#else
#define PRINT_B(...) ((void)0)
#define PRINT_T(...) ((void)0)
#endif

#if XPU_IS_CUDA
#define XPU_CHOOSE(hip, cuda) cuda
#else
#define XPU_CHOOSE(hip, cuda) hip
#endif

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

XPU_D XPU_FORCE_INLINE void barrier() { __syncthreads(); }

namespace detail {

#if XPU_IS_CUDA
namespace cub = ::cub;
#else // XPU_IS_HIP
namespace cub = ::hipcub;
#endif

template<typename T>
struct numeric_limits {};

template<>
struct numeric_limits<float> {
    __device__ static constexpr float max_or_inf() { return INFINITY; }
};

template<>
struct numeric_limits<unsigned int> {
    __device__ static constexpr unsigned int max_or_inf() { return UINT_MAX; }
};

template<>
struct numeric_limits<unsigned long int> {
    __device__ static constexpr unsigned long int max_or_inf() { return UINT64_MAX; }
};

} // namespace detail

template<typename Key, typename T, int BlockSize, int ItemsPerThread>
class block_sort<Key, T, BlockSize, ItemsPerThread, XPU_COMPILATION_TARGET> {

public:
    using block_radix_sort = detail::cub::BlockRadixSort<Key, BlockSize, ItemsPerThread, short>;
    using tempStorage = typename block_radix_sort::TempStorage;
    using key_t = Key;
    using data_t = T;
    using block_merge_t = block_merge<data_t, BlockSize, ItemsPerThread>;
    //using storage_t = typename block_radix_sort::TempStorage;

    union storage_t {
        tempStorage sharedSortMem;
        #ifndef SEQ_MERGE
        typename block_merge_t::storage_t sharedMergeMem;
        #endif
    };

    __device__ block_sort(storage_t &storage_) : storage(storage_) {}

    //KHUN
    template<typename KeyGetter>
    __device__ data_t *sort(data_t *data, size_t N, data_t *buf, KeyGetter &&getKey) {
        return radix_sort(data, N, buf, getKey);
    }

    //     template<typename T, typename KeyGetter>
    // __device__ T *sort(T *data, size_t N, T *buf, int* indices, T* shared_keys, KeyGetter &&getKey) {
    //     return radix_sort(data, N, buf, getKey);
    // }

private:
    storage_t &storage;

    template<typename KeyGetter>
     __device__ data_t *radix_sort(data_t *data, size_t N, data_t *buf, KeyGetter &&getKey) {
        const int ItemsPerBlock = BlockSize * ItemsPerThread;

    // __device__ T *radix_sort(T *data, size_t N, T *buf, KeyGetter &&getKey) {
    //     const int ItemsPerBlock = BlockSize * ItemsPerThread;

        size_t nItemBlocks = N / ItemsPerBlock + (N % ItemsPerBlock > 0 ? 1 : 0);
        Key keys_local[ItemsPerThread];
        short index_local[ItemsPerThread];

        for (size_t i = 0; i < nItemBlocks; i++) {
            size_t start = i * ItemsPerBlock;
            for (size_t b = 0; b < ItemsPerThread; b++) {
                short idx = b * BlockSize + xpu::thread_idx::x();
                size_t global_idx = start + idx;
                if (global_idx < N) {
                    keys_local[b] = getKey(data[global_idx]);
                 } else {
                    keys_local[b] = detail::numeric_limits<Key>::max_or_inf();
                }
                index_local[b] = idx;
            }

            block_radix_sort(storage.sharedSortMem).Sort(keys_local, index_local);

            data_t tmp[ItemsPerThread];

            for (size_t b = 0; b < ItemsPerThread; b++) {
                size_t from = start + index_local[b];
                if (from < N) {
                    tmp[b] = data[from];
                }
            }
            __syncthreads();

            for (size_t b = 0; b < ItemsPerThread; b++) {
                size_t to = start + thread_idx::x() * ItemsPerThread + b;
                if (to < N) {
                    data[to] = tmp[b];
                }
            }
            __syncthreads();

        }

        __syncthreads();

        data_t *src = data;
        data_t *dst = buf;

        for (size_t blockSize = ItemsPerBlock; blockSize < N; blockSize *= 2) {

            size_t carryStart = 0;
            for (size_t st = 0; st + blockSize < N; st += 2 * blockSize) {
                size_t st2 = st + blockSize;
                size_t blockSize2 = min((unsigned long long int)(N - st2), (unsigned long long int)blockSize);
                carryStart = st2 + blockSize2;

                #ifdef SEQ_MERGE
                seq_merge(&src[st], &src[st2], blockSize, blockSize2, &dst[st], getKey);
                #else
                auto comp = [&](const data_t &a, const data_t &b) { return getKey(a) < getKey(b); };
                block_merge_t(storage.sharedMergeMem).merge(&src[st], blockSize, &src[st2], blockSize2, &dst[st], comp);
                #endif
            }

            for (size_t i = carryStart + thread_idx::x(); i < N; i += block_dim::x()) {
                dst[i] = src[i];
            }

            __syncthreads();

            data_t *tmp = src;
            src = dst;
            dst = tmp;
        }

        return src;
    }

    template<typename KeyGetter>
    __device__ void seq_merge(const data_t *block1, const data_t *block2, size_t block_size1, size_t block_size2, data_t *out, KeyGetter &&getKey) {
        if (thread_idx::x() > 0) {
            return;
        }

        size_t i1 = 0;
        size_t i2 = 0;
        size_t i_out = 0;

        while (i1 < block_size1 && i2 < block_size2) {
            if (getKey(block1[i1]) < getKey(block2[i2])) {
                out[i_out] = block1[i1];
                i1++;
            } else {
                out[i_out] = block2[i2];
                i2++;
            }
            i_out++;
        }

        size_t r_i = (i1 < block_size1 ? i1 : i2);
        size_t r_size = (i1 < block_size1 ? block_size1 : block_size2);
        const data_t *r_block = (i1 < block_size1 ? block1 : block2);
        for (; r_i < r_size; r_i++, i_out++) {
            out[i_out] = r_block[r_i];
        }
    }

};

template<typename Key, int BlockSize, int ItemsPerThread>
class block_merge<Key, BlockSize, ItemsPerThread, XPU_COMPILATION_TARGET> {

public:
    using data_t = Key;

    struct storage_t {
        data_t sharedMergeMem[ItemsPerThread * BlockSize + 1];
    };

    XPU_D block_merge(storage_t &storage) : storage(storage) {}

    template<typename Compare>
    XPU_D void merge(const data_t *a, size_t size_a, const data_t *b, size_t size_b, data_t *dst, Compare &&comp) {

        PRINT_B("Merging arrays of size %llu and %llu", size_a, size_b);

        int diag_next = 0;
        int mp_next = merge_path<MgpuBoundsLower>(a, size_a, b, size_b, diag_next, comp);
        for (int diag = 0; diag < size_a + size_b; diag = diag_next) {

            int mp = mp_next;

            diag_next = min(diag_next + BlockSize * ItemsPerThread, size_a + size_b);
            mp_next = merge_path<MgpuBoundsLower>(a, size_a, b, size_b, diag_next, comp);

            int4 range;
            range.x = mp;
            range.y = mp_next;
            range.z = diag - mp;
            range.w = (diag_next - mp_next);

            PRINT_B("Merging [%d, %d] and [%d, %d]", range.x, range.y, range.z, range.w);

            data_t results[ItemsPerThread];

            merge_keys(a, b, range, storage.sharedMergeMem, results, comp);

            thread_to_shared(results, storage.sharedMergeMem, true);

            // Store merged keys to global memory.
            int aCount = range.y - range.x;
            int bCount = range.w - range.z;
            shared_to_global(aCount + bCount, storage.sharedMergeMem, dst+diag, true);
        }
    }

private:
    storage_t &storage;

    /*********************************************** KHUN BEGIN *******************************************************/

    enum MgpuBounds {
        MgpuBoundsLower,
        MgpuBoundsUpper
    };

    template<typename Compare>
    __device__ void merge_keys(const data_t *a, const data_t *b, int4 range, data_t* data_shared, data_t* results, Compare &&comp) {

        int a0 = range.x;
        int a1 = range.y;
        int b0 = range.z;
        int b1 = range.w;

        int aCount = a1 - a0;
        int bCount = b1 - b0;

        // Load the data into shared memory.
        load_to_shared(a + a0, aCount, b + b0, bCount, data_shared, true);

        // Run a merge path to find the start of the serial merge for each thread.
        int diag = ItemsPerThread * thread_idx::x();
        int diag_next = min(diag + ItemsPerThread, aCount + bCount);
        int mp = merge_path<MgpuBoundsLower>(data_shared, aCount, data_shared + aCount, bCount, diag, comp);
        int mp_next = merge_path<MgpuBoundsLower>(data_shared, aCount, data_shared + aCount, bCount, diag_next, comp);

        // Compute the ranges of the sources in shared memory.
        int a0tid = mp;
        int a1tid = mp_next;
        int b0tid = aCount + diag - mp;
        int b1tid = aCount + diag_next - mp_next;

        // PRINT_T("merge path: a0 = %d, a1 = %d, b0 = %d, b1 = %d", a0tid, a1tid, b0tid, b1tid);

        // Serial merge into register.
        serial_merge(data_shared, a0tid, a1tid, b0tid, b1tid, results, comp);
    }

    template<typename Compare>
    __device__ void serial_merge(const data_t* keys_shared, int aBegin,  int aEnd, int bBegin, int bEnd, data_t* results, Compare &&comp) {
        data_t aKey = keys_shared[aBegin];
        data_t bKey = keys_shared[bBegin];

        constexpr bool range_check = true;

        // PRINT_T("Merging from SMEM, a0 = %d, a1 = %d, b0 = %d, b1 = %d", aBegin, aEnd, bBegin, bEnd);

        #pragma unroll
        for (int i = 0; i < ItemsPerThread && (bBegin < bEnd || aBegin < aEnd); ++i) {
            bool p;
            if (range_check) {
                p = (bBegin >= bEnd) || ((aBegin < aEnd) && !comp(bKey, aKey));
            } else {
                p = !comp(bKey, aKey);
            }

            results[i] = p ? aKey : bKey;

            // assert(aBegin < BlockSize * ItemsPerThread);
            // assert(bBegin < BlockSize * ItemsPerThread);

            // PRINT_T("aBegin = %d, bBegin = %d", aBegin, bBegin);

            if (p) {
                aKey = keys_shared[++aBegin];
            } else {
                bKey = keys_shared[++bBegin];
            }
        }
        __syncthreads();
    }

    template<MgpuBounds Bounds, typename Compare>
    __device__ int merge_path(const data_t *a, int aCount, const data_t *b, int bCount, int diag, Compare &&comp) {

        PRINT_B("Merge path for diag %d", diag);

        int begin = max(0, diag - bCount);
        int end = min(diag, aCount);

        PRINT_B("begin = %d, end = %d", begin, end);

        while (begin < end) {
            int mid = (begin + end) >> 1;
            data_t aKey = a[mid];
            // printf("tid %d: bidx = %d\n", thread_idx::x(), diag-1-mid);

            PRINT_T("b_idx = %d", diag-1-mid);
            data_t bKey = b[diag - 1 - mid];
            bool pred = (MgpuBoundsUpper == Bounds) ? comp(aKey, bKey) : !comp(bKey, aKey);

            PRINT_B("aKey = %f, bKey = %f, begin = %d, end = %d", aKey, bKey, begin, end);

            if (pred) {
                begin = mid + 1;
            } else {
                end = mid;
            }
        }
        return begin;
    }

    __device__ void reg_to_shared(const data_t* reg, data_t *dest, bool sync) {

        #pragma unroll
        for(int i = 0; i < ItemsPerThread; ++i) {
            assert(BlockSize * i + thread_idx::x() < BlockSize * ItemsPerThread);
            dest[BlockSize * i + thread_idx::x()] = reg[i];
        }

        if(sync) {
            __syncthreads();
        }
    }

    __device__ void load_to_reg(const data_t *a_global, int aCount, const data_t *b_global, int bCount, data_t *reg, bool sync)  {

        b_global -= aCount;
        int total = aCount + bCount;

        #pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i) {
            int index = BlockSize * i + thread_idx::x();
            if (index < aCount) reg[i] = a_global[index];
            else if (index < total) reg[i] = b_global[index];
        }

        if (sync) {
            __syncthreads();
        }
    }

    __device__ void load_to_shared(const data_t *a_global, int aCount, const data_t *b_global, int bCount, data_t* shared, bool sync) {
        data_t reg[ItemsPerThread];
        load_to_reg(a_global, aCount, b_global, bCount, reg, false);
        reg_to_shared(reg,  shared, sync);
    }

    __device__ void thread_to_shared(const data_t* threadReg, data_t* shared, bool sync) {
        #pragma unroll
        for(int i = 0; i < ItemsPerThread; ++i) {
            assert(ItemsPerThread * thread_idx::x() + i < BlockSize * ItemsPerThread);
            shared[ItemsPerThread * thread_idx::x() + i] = threadReg[i];
        }

        if(sync) {
            __syncthreads();
        }
    }

    __device__ void shared_to_global(int count, const data_t* source, data_t *dest, bool sync) {
        #pragma unroll
        for(int i = 0; i < ItemsPerThread; ++i) {
            int index = BlockSize * i + thread_idx::x();
            if(index < count) {
                dest[index] = source[index];
            }
        }
        if(sync) __syncthreads();
    }

    /************************************************ KHUN END ********************************************************/
};

} // namespace xpu

#endif
