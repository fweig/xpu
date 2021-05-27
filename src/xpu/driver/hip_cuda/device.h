
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

} // namespace detail

template<typename C> XPU_D XPU_FORCE_INLINE const C &cmem() { return cmem_accessor<C>::get(); }

template<typename Key, typename KeyValueType, int BlockSize, int ItemsPerThread>
class block_sort<Key, KeyValueType, BlockSize, ItemsPerThread, xpu::driver::cuda> {

public:
    using block_radix_sort = cub::BlockRadixSort<Key, BlockSize, ItemsPerThread, short>;
    using tempStorage = typename block_radix_sort::TempStorage;
    //using storage_t = typename block_radix_sort::TempStorage;

    struct storage_t{
        tempStorage sharedSortMem;
        KeyValueType sharedMergeMem[1000];
        int indices[1000];
    };

    __device__ block_sort(storage_t &storage_) : storage(storage_) {}

    //KHUN
    template<typename KeyGetter>
    __device__ KeyValueType *sort(KeyValueType *data, size_t N, KeyValueType *buf, KeyGetter &&getKey) {
        return radix_sort(data, N, buf, getKey);
    }

    //     template<typename T, typename KeyGetter>
    // __device__ T *sort(T *data, size_t N, T *buf, int* indices, T* shared_keys, KeyGetter &&getKey) {
    //     return radix_sort(data, N, buf, getKey);
    // }

private:
    storage_t &storage;



    template<typename KeyGetter>
     __device__ KeyValueType *radix_sort(KeyValueType *data, size_t N, KeyValueType *buf, KeyGetter &&getKey) {
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

            KeyValueType tmp[ItemsPerThread];

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

        KeyValueType *src = data;
        KeyValueType *dst = buf;

        for (size_t blockSize = ItemsPerBlock; blockSize < N; blockSize *= 2) {

            size_t carryStart = 0;
            for (size_t st = 0; st + blockSize < N; st += 2 * blockSize) {
                size_t st2 = st + blockSize;
                size_t blockSize2 = min((unsigned long long int)(N - st2), (unsigned long long int)blockSize);
                carryStart = st2 + blockSize2;

                //merge(&src[st], &src[st2], blockSize, blockSize2, &dst[st], getKey);
                constexpr int vt_var = 4;
                KeyValueType results[vt_var];
                int indices[vt_var];

                int4 range;
                range.x = st;
                range.y = st + blockSize;
                range.z = st2;
                range.w = st2 + blockSize2;
                //int cnt = 10;
                //int sze = blockSize;

                // printf("%d range.x: %d \n", threadIdx.x, range.x);
                // printf("%d range.y: %d \n", threadIdx.x, range.y);
                // printf("%d blockSize: %d \n", threadIdx.x, range.y-range.x);
                // int indices[32];
                // KeyValueType *strg;

                DeviceMergeKeysIndices<BlockSize,vt_var>(src, range, threadIdx.x, storage.sharedMergeMem, results, indices, [&](KeyValueType &tOne, KeyValueType  &tTwo){ return (getKey(tOne) >= getKey(tTwo));} );

                DeviceThreadToShared<vt_var>(results, threadIdx.x, storage.sharedMergeMem, true);

                // Store merged keys to global memory.
                int aCount = range.y - range.x;
                int bCount = range.w - range.z;
                DeviceSharedToGlobal<BlockSize, vt_var>(aCount + bCount, storage.sharedMergeMem, threadIdx.x, &dst[st] + BlockSize * vt_var * 0, true);

                //printf("%d blockSizeInLoop: %d \n", threadIdx.x, blockSize); = 64
                // if(true){
                //     // for(int i = st; i < sze; i++){
                //     //     printf("%d, ", i);
                //     // }
                //     for(int i = st; i < sze; i++){
                //         auto arr = getKey(dst[i]) <= getKey(dst[i+1]);
                //         // printf("%d, ", dst[i]);
                //         // if(i == 10){
                //         //     printf("\n");
                //         // }
                //         if(arr == false){
                //             printf("%d ErrorPosition: %d -> %d > %d \n", threadIdx.x, i, getKey(dst[i]), getKey(dst[i+1]));
                //             break;
                //         }
                //     }
                // }

            }

            for (size_t i = carryStart + thread_idx::x(); i < N; i += block_dim::x()) {
                dst[i] = src[i];
            }

            __syncthreads();

            KeyValueType *tmp = src;
            src = dst;
            dst = tmp;
        }

        return src;
    }

    /*********************************************** KHUN BEGIN *******************************************************/

    enum MgpuBounds {
        MgpuBoundsLower,
        MgpuBoundsUpper
    };

    template<typename T>
    struct DevicePair {
        T x, y;
    };


    template<int NT, int VT, typename It1, typename Compare>
    __device__ void DeviceMergeKeysIndices(It1 data, int4 range,
    int tid, KeyValueType* keys_shared, KeyValueType* results, int *indices, Compare &&comp) {

        int a0 = range.x;
        int a1 = range.y;
        int b0 = range.z;
        int b1 = range.w;
        int aCount = a1 - a0;
        int bCount = b1 - b0;

        // Load the data into shared memory.
        DeviceLoad2ToShared<NT, VT, VT+1>(data + a0, aCount, data + b0,
            bCount, tid, keys_shared,true);
        // Run a merge path to find the start of the serial merge for each thread.
        int diag = VT * tid;
        int mp = MergePath<MgpuBoundsLower>(keys_shared, aCount,
            keys_shared + aCount, bCount, diag, comp);

        // int mp = MergePath<MgpuBoundsLower>(keys_shared, aCount,
        //     keys_shared + aCount, bCount, diag, comp);

        int mp = MergePath<MgpuBoundsLower>(a_global, aCount,
            a_global + aCount, bCount, diag, comp);

        // Compute the ranges of the sources in shared memory.
        int a0tid = mp;
        int a1tid = aCount;
        int b0tid = aCount + diag - mp;
        int b1tid = aCount + bCount;
        // Serial merge into register.
        SerialMerge<VT, true>(keys_shared, a0tid, a1tid, b0tid, b1tid, results,
            indices, comp);

        // if(threadIdx.x==0){
        //     auto errFound = false;
        //     //printf("%d aBegin: %d, aEnd %d, Range: %d\n", threadIdx.x, a0tid, a1tid, a1tid-a0tid);
        //     for(int i = 0; i< range.y - range.x; i++){
        //         if(!comp(results[i+1], results[i])){
        //             errFound = true;
        //             printf("Last ErrorPosition: %d\n", i);
        //         }
        //     }
        //     // for(int i = 0; i < 999; i++){
        //     //     if(!comp(keys_shared[indices[i+1]], keys_shared[indices[i]])){
        //     //         errFound = true;
        //     //         printf("Last ErrorPosition: %d\n", i);
        //     //     }
        //     // }

        //     if(!errFound){
        //         printf("No Error \n");
        //     }
        // }
    }

    template<int VT, bool RangeCheck, typename Compare>
    __device__ void SerialMerge(const KeyValueType* keys_shared, int aBegin,  int aEnd,
                                 int bBegin, int bEnd, KeyValueType* results, int* indices, Compare &&comp) {



        KeyValueType aKey = keys_shared[aBegin];
        KeyValueType bKey = keys_shared[bBegin];
        #pragma unroll
        for(int i = 0; i < VT; ++i) {
            bool p;
            if(RangeCheck)
                p = (bBegin >= bEnd) || ((aBegin < aEnd) && !comp(bKey, aKey));
            else
                p = !comp(bKey, aKey);

            results[i] = p ? aKey : bKey;
            indices[i] = p ? aBegin : bBegin;

            if(p) aKey = keys_shared[++aBegin];
            else bKey = keys_shared[++bBegin];
        }
        __syncthreads();


    }


    template<int NT, int VT, typename OutputIt>
    __device__ void DeviceRegToShared(const KeyValueType* reg, int tid,
                                       OutputIt dest, bool sync) {

        typedef typename std::iterator_traits<OutputIt>::value_type T2;
        #pragma unroll
        for(int i = 0; i < VT; ++i)
            dest[NT * i + tid] = (T2)reg[i];

        if(sync) __syncthreads();
    }


    template<MgpuBounds Bounds, typename It1, typename It2, typename Compare>
    __device__ int MergePath(It1 a, int aCount, It2 b, int bCount, int diag,
                                   Compare &&comp) {

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
            // bool pred = (MgpuBoundsUpper == Bounds) ?
            //              (getKey(aKey) <= getKey(bKey)) :
            //              !(getKey(bKey) <= getKey(aKey));
            if(pred) begin = mid + 1;
            else end = mid;
        }
        return begin;
    }


        template<int NT, int VT0, int VT1, typename InputIt1, typename InputIt2>
    __device__ void DeviceLoad2ToReg(InputIt1 a_global, int aCount,
                                      InputIt2 b_global, int bCount, int tid, KeyValueType* reg, bool sync)  {

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

    template<int NT, int VT0, int VT1, typename InputIt1, typename InputIt2>
    __device__ void DeviceLoad2ToShared(InputIt1 a_global, int aCount,
                                         InputIt2 b_global, int bCount, int tid, KeyValueType* shared, bool sync) {

        KeyValueType reg[VT1];
        DeviceLoad2ToReg<NT, VT0, VT1>(a_global, aCount, b_global, bCount, tid,
                                       reg, false);
        DeviceRegToShared<NT, VT1>(reg, tid, shared, sync);
    }

        template<int VT, typename T>
    __device__ void DeviceThreadToShared(const T* threadReg, int tid, T* shared,
        bool sync) {

        if(1 & VT) {
            // Odd grain size. Store as type T.
            #pragma unroll
            for(int i = 0; i < VT; ++i)
                shared[VT * tid + i] = threadReg[i];
        } else {
            // Even grain size. Store as DevicePair<T>. This lets us exploit the
            // 8-byte shared memory mode on Kepler.
            DevicePair<T>* dest = (DevicePair<T>*)(shared + VT * tid);
            #pragma unroll
            for(int i = 0; i < VT / 2; ++i)
                dest[i] = MakeDevicePair(threadReg[2 * i], threadReg[2 * i + 1]);
        }
        if(sync) __syncthreads();
    }

    template<typename T>
    __device__ DevicePair<T> MakeDevicePair(T x, T y) {
	    DevicePair<T> p = { x, y };
	    return p;
    }

    template<int NT, int VT, typename T, typename OutputIt>
    __device__ void DeviceSharedToGlobal(int count, const T* source, int tid,
        OutputIt dest, bool sync) {

        typedef typename std::iterator_traits<OutputIt>::value_type T2;
        #pragma unroll
        for(int i = 0; i < VT; ++i) {
            int index = NT * i + tid;
            if(index < count) dest[index] = (T2)source[index];
        }
        if(sync) __syncthreads();
    }


    /************************************************ KHUN END ********************************************************/



    template<typename KeyGetter>
    __device__ void merge(const KeyValueType *block1, const KeyValueType *block2, size_t block_size1, size_t block_size2, KeyValueType *out, KeyGetter &&getKey) {
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
        const KeyValueType *r_block = (i1 < block_size1 ? block1 : block2);
        for (; r_i < r_size; r_i++, i_out++) {
            out[i_out] = r_block[r_i];
        }
    }




};



} // namespace xpu

#endif
