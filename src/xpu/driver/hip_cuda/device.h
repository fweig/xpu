
#ifndef XPU_DRIVER_CUDA_DEVICE_RUNTIME_H
#define XPU_DRIVER_CUDA_DEVICE_RUNTIME_H

#ifndef XPU_DEVICE_H
#error "This header should not be included directly. Include xpu/device.h instead."
#endif

#include "../../detail/constant_memory.h"
#include "../../detail/macros.h"

#if XPU_IS_CUDA
#include <cub/cub.cuh>
#elif XPU_IS_HIP
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#else
#error "Internal XPU error: This should never happen."
#endif

#include <iostream>
#include <type_traits>

#if 0
#define PRINT_B(message, ...) if (xpu::pos.thread_idx_x() == 0) printf("t 0: " message "\n", ##__VA_ARGS__)
#define PRINT_T(message, ...) do { printf("t %d: " message "\n", xpu::pos.thread_idx_x(), ##__VA_ARGS__); } while (0)
#else
#define PRINT_B(...) ((void)0)
#define PRINT_T(...) ((void)0)
#endif

#define XPU_DETAIL_ASSERT(x) static_cast<void>(0)

namespace xpu {

XPU_D XPU_FORCE_INLINE int abs(int a) { return ::abs(a); }
XPU_D XPU_FORCE_INLINE float abs(float x) { return ::fabsf(x); }
XPU_D XPU_FORCE_INLINE float acos(float x) { return ::acosf(x); }
XPU_D XPU_FORCE_INLINE float acosh(float x) { return ::acoshf(x); }
XPU_D XPU_FORCE_INLINE float asin(float x) { return ::asinf(x); }
XPU_D XPU_FORCE_INLINE float asinh(float x) { return ::asinhf(x); }
XPU_D XPU_FORCE_INLINE float atan2(float y, float x) { return ::atan2f(y, x); }
XPU_D XPU_FORCE_INLINE float atan(float x) { return ::atanf(x); }
XPU_D XPU_FORCE_INLINE float atanh(float x) { return ::atanhf(x); }
XPU_D XPU_FORCE_INLINE float cbrt(float x) { return ::cbrtf(x); }
XPU_D XPU_FORCE_INLINE float ceil(float x) { return ::ceilf(x); }
XPU_D XPU_FORCE_INLINE float copysign(float x, float y) { return ::copysignf(x, y); }
XPU_D XPU_FORCE_INLINE float cos(float x) { return ::cosf(x); }
XPU_D XPU_FORCE_INLINE float cosh(float x) { return ::coshf(x); }
XPU_D XPU_FORCE_INLINE float cospi(float x) { return ::cospi(x); }
XPU_D XPU_FORCE_INLINE float erf(float x) { return ::erff(x); }
XPU_D XPU_FORCE_INLINE float erfc(float x) { return ::erfcf(x); }
XPU_D XPU_FORCE_INLINE float exp2(float x) { return ::exp2f(x); }
XPU_D XPU_FORCE_INLINE float exp(float x) { return ::expf(x); }
XPU_D XPU_FORCE_INLINE float expm1(float x) { return ::expm1(x); }
XPU_D XPU_FORCE_INLINE float fdim(float x, float y) { return ::fdimf(x, y); }
XPU_D XPU_FORCE_INLINE float floor(float x) { return ::floorf(x); }
XPU_D XPU_FORCE_INLINE float fma(float x, float y, float z) { return ::fmaf(x, y, z); }
XPU_D XPU_FORCE_INLINE float fmod(float x, float y) { return ::fmodf(x, y); }
XPU_D XPU_FORCE_INLINE float hypot(float x, float y) { return ::hypotf(x, y); }
XPU_D XPU_FORCE_INLINE int ilogb(float x) { return ::ilogbf(x); }
XPU_D XPU_FORCE_INLINE bool isfinite(float a) { return ::isfinite(a); }
XPU_D XPU_FORCE_INLINE bool isinf(float x) { return ::isinf(x); }
XPU_D XPU_FORCE_INLINE bool isnan(float x) { return ::isnan(x); }
XPU_D XPU_FORCE_INLINE float j0(float x) { return ::j0f(x); }
XPU_D XPU_FORCE_INLINE float j1(float x) { return ::j1f(x); }
XPU_D XPU_FORCE_INLINE float jn(int n, float x) { return ::jnf(n, x); }
XPU_D XPU_FORCE_INLINE float ldexp(float x, int exp) { return ::ldexpf(x, exp); }
XPU_D XPU_FORCE_INLINE long long int llrint(float x) { return ::llrintf(x); }
XPU_D XPU_FORCE_INLINE long long int llround(float x) { return ::llroundf(x); }
XPU_D XPU_FORCE_INLINE float log(float x) { return ::logf(x); }
XPU_D XPU_FORCE_INLINE float log10(float x) { return ::log10f(x); }
XPU_D XPU_FORCE_INLINE float log1p(float x) { return ::log1pf(x); }
XPU_D XPU_FORCE_INLINE float log2(float x) { return ::log2f(x); }
XPU_D XPU_FORCE_INLINE float logb(float x) { return ::logbf(x); }
XPU_D XPU_FORCE_INLINE long int lrint(float x) { return ::lrintf(x); }
XPU_D XPU_FORCE_INLINE long int lround(float x) { return ::lroundf(x); }
XPU_D XPU_FORCE_INLINE int max(int a, int b) { return ::max(a, b); }
XPU_D XPU_FORCE_INLINE unsigned int max(unsigned int a, unsigned int b) { return ::max(a, b); }
XPU_D XPU_FORCE_INLINE long long int max(long long int a, long long int b) { return ::max(a, b); }
XPU_D XPU_FORCE_INLINE unsigned long long int max(unsigned long long int a, unsigned long long int b) { return ::max(a, b); }
XPU_D XPU_FORCE_INLINE float max(float a, float b) { return ::fmaxf(a, b); }
XPU_D XPU_FORCE_INLINE int min(int a, int b) { return ::min(a, b); }
XPU_D XPU_FORCE_INLINE unsigned int min(unsigned int a, unsigned int b) { return ::min(a, b); }
XPU_D XPU_FORCE_INLINE long long int min(long long int a, long long int b) { return ::min(a, b); }
XPU_D XPU_FORCE_INLINE unsigned long long int min(unsigned long long int a, unsigned long long int b) { return ::min(a, b); }
XPU_D XPU_FORCE_INLINE float min(float a, float b) { return ::fminf(a, b); }
XPU_D XPU_FORCE_INLINE float nan(const char *tagp) { return ::nanf(tagp); }
XPU_D XPU_FORCE_INLINE float nearbyint(float x) { return ::nearbyint(x); }
XPU_D XPU_FORCE_INLINE float norm(int dim, const float *a) { return ::normf(dim, a); }
XPU_D XPU_FORCE_INLINE float norm3d(float a, float b, float c) { return ::norm3df(a, b, c); }
XPU_D XPU_FORCE_INLINE float norm4d(float a, float b, float c, float d) { return ::norm4df(a, b, c, d); }
// XPU_D XPU_FORCE_INLINE float normcdf(float y) { return ::normcdff(y); }
// XPU_D XPU_FORCE_INLINE float normcdfinv(float y) { return ::normcdfinvf(y); }
XPU_D XPU_FORCE_INLINE float pow(float x, float y) { return ::powf(x, y); }
XPU_D XPU_FORCE_INLINE float rcbrt(float x) { return ::rcbrtf(x); }
XPU_D XPU_FORCE_INLINE float remainder(float x, float y) { return ::remainderf(x, y); }
XPU_D XPU_FORCE_INLINE float remquo(float x, float y, int *quo) { return ::remquof(x, y, quo); }
XPU_D XPU_FORCE_INLINE float rhypot(float x, float y) { return ::rhypotf(x, y); }
XPU_D XPU_FORCE_INLINE float rint(float x) { return ::rintf(x); }
XPU_D XPU_FORCE_INLINE float rnorm(int dim, const float *a) { return ::rnormf(dim, a); }
XPU_D XPU_FORCE_INLINE float rnorm3d(float a, float b, float c) { return ::rnorm3df(a, b, c); }
XPU_D XPU_FORCE_INLINE float rnorm4d(float a, float b, float c, float d) { return ::rnorm4df(a, b, c, d); }
XPU_D XPU_FORCE_INLINE float round(float x) { return ::roundf(x); }
XPU_D XPU_FORCE_INLINE float rsqrt(float x) { return ::rsqrtf(x); }
XPU_D XPU_FORCE_INLINE float scalbln(float x, long int n) { return ::scalblnf(x, n); }
XPU_D XPU_FORCE_INLINE float scalbn(float x, int n) { return ::scalbnf(x, n); }
XPU_D XPU_FORCE_INLINE bool signbit(float a) { return ::signbit(a); }
XPU_D XPU_FORCE_INLINE void sincos(float x, float *sptr, float *cptr) { return ::sincosf(x, sptr, cptr); }
XPU_D XPU_FORCE_INLINE void sincospi(float x, float *sptr, float *cptr) { return ::sincospif(x, sptr, cptr); }
XPU_D XPU_FORCE_INLINE float sin(float x) { return ::sinf(x); }
XPU_D XPU_FORCE_INLINE float sinh(float x) { return ::sinhf(x); }
XPU_D XPU_FORCE_INLINE float sinpi(float x) { return ::sinpif(x); }
XPU_D XPU_FORCE_INLINE float sqrt(float x) { return ::sqrtf(x); }
XPU_D XPU_FORCE_INLINE float tan(float x) { return ::tanf(x); }
XPU_D XPU_FORCE_INLINE float tanh(float x) { return ::tanhf(x); }
XPU_D XPU_FORCE_INLINE float tgamma(float x) { return ::tgammaf(x); }
XPU_D XPU_FORCE_INLINE float trunc(float x) { return ::truncf(x); }
XPU_D XPU_FORCE_INLINE float y0(float x) { return ::y0f(x); }
XPU_D XPU_FORCE_INLINE float y1(float x) { return ::y1f(x); }
XPU_D XPU_FORCE_INLINE float yn(int n, float x) { return ::ynf(n, x); }

XPU_D XPU_FORCE_INLINE int atomic_cas(int *addr, int compare, int val) {
    return atomicCAS(addr, compare, val);
}

XPU_D XPU_FORCE_INLINE unsigned int atomic_cas(unsigned int *addr, unsigned int compare, unsigned int val) {
    return atomicCAS(addr, compare, val);
}

XPU_D XPU_FORCE_INLINE float atomic_cas(float *addr, float compare, float val) {
    return __int_as_float(atomicCAS((int *) addr, __float_as_int(compare), __float_as_int(val)));
}

XPU_D XPU_FORCE_INLINE int atomic_cas_block(int *addr, int compare, int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicCAS_block(addr, compare, val);
#else
    return atomicCAS(addr, compare, val);
#endif
}

XPU_D XPU_FORCE_INLINE unsigned int atomic_cas_block(unsigned int *addr, unsigned int compare, unsigned int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicCAS_block(addr, compare, val);
#else
    return atomicCAS(addr, compare, val);
#endif
}

XPU_D XPU_FORCE_INLINE float atomic_cas_block(float *addr, float compare, float val) {
    return int_as_float(atomic_cas_block((int *) addr, float_as_int(compare), float_as_int(val)));
}

XPU_D XPU_FORCE_INLINE int atomic_add(int *addr, int compare, int val) {
    return atomicAdd(addr, val);
}

XPU_D XPU_FORCE_INLINE unsigned int atomic_add(unsigned int *addr, unsigned int compare, unsigned int val) {
    return atomicAdd(addr, val);
}

XPU_D XPU_FORCE_INLINE float atomic_add(float *addr, float val) {
    return atomicAdd(addr, val);
}

XPU_D XPU_FORCE_INLINE int atomic_add_block(int *addr, int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicAdd_block(addr, val);
#else
    return atomicAdd(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE unsigned int atomic_add_block(unsigned int *addr, unsigned int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicAdd_block(addr, val);
#else
    return atomicAdd(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE float atomic_add_block(float *addr, float val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicAdd_block(addr, val);
#else
    return atomicAdd(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE int atomic_sub(int *addr, int val) {
    return atomicSub(addr, val);
}

XPU_D XPU_FORCE_INLINE unsigned int atomic_sub(unsigned int *addr, unsigned int val) {
    return atomicSub(addr, val);
}

XPU_D XPU_FORCE_INLINE int atomic_sub_block(int *addr, int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicSub_block(addr, val);
#else
    return atomicSub(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE unsigned int atomic_sub_block(unsigned int *addr, unsigned int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicSub_block(addr, val);
#else
    return atomicSub(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE int atomic_and(int *addr, int val) {
    return atomicAnd(addr, val);
}

XPU_D XPU_FORCE_INLINE unsigned int atomic_and(unsigned int *addr, unsigned int val) {
    return atomicAnd(addr, val);
}

XPU_D XPU_FORCE_INLINE int atomic_and_block(int *addr, int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicAnd_block(addr, val);
#else
    return atomicAnd(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE unsigned int atomic_and_block(unsigned int *addr, unsigned int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicAnd_block(addr, val);
#else
    return atomicAnd(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE int atomic_or(int *addr, int val) {
    return atomicOr(addr, val);
}

XPU_D XPU_FORCE_INLINE unsigned int atomic_or(unsigned int *addr, unsigned int val) {
    return atomicOr(addr, val);
}

XPU_D XPU_FORCE_INLINE int atomic_or_block(int *addr, int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicOr_block(addr, val);
#else
    return atomicOr(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE unsigned int atomic_or_block(unsigned int *addr, unsigned int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicOr_block(addr, val);
#else
    return atomicOr(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE int atomic_xor(int *addr, int val) {
    return atomicXor(addr, val);
}

XPU_D XPU_FORCE_INLINE unsigned int atomic_xor(unsigned int *addr, unsigned int val) {
    return atomicXor(addr, val);
}

XPU_D XPU_FORCE_INLINE int atomic_xor_block(int *addr, int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicXor_block(addr, val);
#else
    return atomicXor(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE unsigned int atomic_xor_block(unsigned int *addr, unsigned int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicXor_block(addr, val);
#else
    return atomicXor(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE void barrier() { __syncthreads(); }

XPU_D XPU_FORCE_INLINE int float_as_int(float val) { return __float_as_int(val); }
XPU_D XPU_FORCE_INLINE float int_as_float(int val) { return __int_as_float(val); }

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

template<typename T, int BlockSize>
class block_scan<T, BlockSize, XPU_COMPILATION_TARGET> {

private:
    using block_scan_impl = detail::cub::BlockScan<T, BlockSize>;

public:
    struct storage_t {
        typename block_scan_impl::TempStorage scanTemp;
    };

    XPU_D block_scan(storage_t &st) : impl(st.scanTemp) {}

    XPU_D void exclusive_sum(T input, T &output) { impl.ExclusiveSum(input, output); }

    template<typename ScanOp>
    XPU_D void exclusive_sum(T input, T &output, T initial_value, ScanOp scan_op) {
        impl.ExclusiveSum(input, output, initial_value, scan_op);
    }

    template<int ItemsPerThread>
    XPU_D void exclusive_sum(T(&input)[ItemsPerThread], T(&output)[ItemsPerThread]) { impl.ExclusiveSum(input, output); }

    template<int ItemsPerThread, typename ScanOp>
    XPU_D void exclusive_sum(T(&input)[ItemsPerThread], T(&output)[ItemsPerThread], T initial_value, ScanOp scan_op) {
        impl.ExclusiveSum(input, output, initial_value, scan_op);
    }

    XPU_D void inclusive_sum(T input, T &output) { impl.InclusiveSum(input, output); }

    template<typename ScanOp>
    XPU_D void inclusive_sum(T input, T &output, T initial_value, ScanOp scan_op) {
        impl.InclusiveSum(input, output, initial_value, scan_op);
    }

    template<int ItemsPerThread>
    XPU_D void inclusive_sum(T(&input)[ItemsPerThread], T(&output)[ItemsPerThread]) { impl.InclusiveSum(input, output); }

    template<int ItemsPerThread, typename ScanOp>
    XPU_D void inclusive_sum(T(&input)[ItemsPerThread], T(&output)[ItemsPerThread], T initial_value, ScanOp scan_op) {
        impl.InclusiveSum(input, output, initial_value, scan_op);
    }

private:
    block_scan_impl impl;

};

template<typename Key, typename T, int BlockSize, int ItemsPerThread>
class block_sort<Key, T, BlockSize, ItemsPerThread, XPU_COMPILATION_TARGET> {

public:
    using block_radix_sort = detail::cub::BlockRadixSort<Key, BlockSize, ItemsPerThread, short>;
    using tempStorage = typename block_radix_sort::TempStorage;
    using key_t = Key;
    using data_t = T;
    using block_merge_t = block_merge<data_t, BlockSize, ItemsPerThread>;
    //using storage_t = typename block_radix_sort::TempStorage;

    static_assert(std::is_trivially_constructible_v<data_t>, "Sorted type needs trivial constructor.");

    union storage_t {
        storage_t() = default;
        tempStorage sharedSortMem;
        #ifndef SEQ_MERGE
        typename block_merge_t::storage_t sharedMergeMem;
        #endif
    };

    __device__ block_sort(tpos &position, storage_t &storage_) : pos(position), storage(storage_) {}

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
    tpos &pos;
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
                short idx = b * BlockSize + pos.thread_idx_x();;
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
                size_t to = start + pos.thread_idx_x() * ItemsPerThread + b;
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
                block_merge_t(pos, storage.sharedMergeMem).merge(&src[st], blockSize, &src[st2], blockSize2, &dst[st], comp);
                #endif
            }

            for (size_t i = carryStart + pos.thread_idx_x(); i < N; i += pos.block_dim_x()) {
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
        if (pos.thread_idx_x() > 0) {
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

    static_assert(std::is_trivially_constructible_v<data_t>, "Merged type needs trivial constructor.");

    struct storage_t {
        data_t sharedMergeMem[ItemsPerThread * BlockSize + 1];
    };

    XPU_D block_merge(tpos& pos, storage_t &storage) : pos(pos), storage(storage) {}

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
    tpos &pos;
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
        int diag = ItemsPerThread * pos.thread_idx_x();
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
            // printf("tid %d: bidx = %d\n", pos.thread_idx_x(), diag-1-mid);

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
            assert(BlockSize * i + pos.thread_idx_x() < BlockSize * ItemsPerThread);
            dest[BlockSize * i + pos.thread_idx_x()] = reg[i];
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
            int index = BlockSize * i + pos.thread_idx_x();
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
            assert(ItemsPerThread * pos.thread_idx_x() + i < BlockSize * ItemsPerThread);
            shared[ItemsPerThread * pos.thread_idx_x() + i] = threadReg[i];
        }

        if(sync) {
            __syncthreads();
        }
    }

    __device__ void shared_to_global(int count, const data_t* source, data_t *dest, bool sync) {
        #pragma unroll
        for(int i = 0; i < ItemsPerThread; ++i) {
            int index = BlockSize * i + pos.thread_idx_x();
            if(index < count) {
                dest[index] = source[index];
            }
        }
        if(sync) __syncthreads();
    }

    /************************************************ KHUN END ********************************************************/
};

} // namespace xpu

#define SAFE_CALL(call) if (int err = call; err != 0) return err
#define ON_ERROR_GOTO(errvar, call, label) errvar = call; \
    if (errvar != 0) goto label

namespace xpu::detail {

template<typename F, typename S, typename... Args>
__global__ void kernel_entry(Args... args) {
    using shared_memory = typename F::shared_memory;
    using constants = typename F::constants;
    using context = kernel_context<shared_memory, constants>;
    __shared__ shared_memory smem;
    tpos pos{internal_ctor};
    F{}(context{pos, smem}, args...);
}

template<typename F, int MaxThreadsPerBlock, typename... Args>
__global__ void __launch_bounds__(MaxThreadsPerBlock) kernel_entry_bounded(Args... args) {
    using shared_memory = typename F::shared_memory;
    using constants = typename F::constants;
    using context = kernel_context<shared_memory, constants>;
    __shared__ shared_memory smem;
    tpos pos{internal_ctor};
    constants cmem{internal_ctor};
    context ctx{internal_ctor, pos, smem, cmem};
    F{}(ctx, args...);
}

#if XPU_IS_CUDA

template<typename F>
struct action_runner<constant_tag, F> {
    using data_t = typename F::data_t;
    static int call(const data_t &val) {
        return cudaMemcpyToSymbol(constant_memory<F>, &val, sizeof(data_t));
    }
};

template<typename K, typename... Args>
struct action_runner<kernel_tag, K, void(K::*)(kernel_context<typename K::shared_memory, typename K::constants> &, Args...)> {

    static int call(float *ms, driver_interface * /*cuda_driver*/, grid g, Args... args) {
        dim block_dim = K::block_size::value;
        dim grid_dim{};

        g.get_compute_grid(block_dim, grid_dim);

        XPU_LOG("Calling kernel '%s' [block_dim = (%d, %d, %d), grid_dim = (%d, %d, %d)] with CUDA driver.", type_name<K>(), block_dim.x, block_dim.y, block_dim.z, grid_dim.x, grid_dim.y, grid_dim.z);

        bool measure_time = (ms != nullptr);
        cudaEvent_t start, end;
        int err = 0;

        if (measure_time) {
            SAFE_CALL(cudaEventCreate(&start));
            ON_ERROR_GOTO(err, cudaEventCreate(&end), cleanup_start_event);
        }

        if (measure_time) {
            ON_ERROR_GOTO(err, cudaEventRecord(start), cleanup_events);
        }

        kernel_entry_bounded<K, K::block_size::value.linear(), Args...><<<grid_dim.as_cuda_grid(), block_dim.as_cuda_grid()>>>(args...);

        if (measure_time) {
            ON_ERROR_GOTO(err, cudaEventRecord(end), cleanup_events);
        }
        SAFE_CALL(cudaDeviceSynchronize());

        if (measure_time) {
            ON_ERROR_GOTO(err, cudaEventSynchronize(end), cleanup_events);
            ON_ERROR_GOTO(err, cudaEventElapsedTime(ms, start, end), cleanup_events);
            XPU_LOG("Kernel '%s' took %f ms", type_name<K>(), *ms);
        }

    cleanup_events:
        if (measure_time) {
            SAFE_CALL(cudaEventDestroy(end));
        }
    cleanup_start_event:
        if (measure_time) {
            SAFE_CALL(cudaEventDestroy(start));
        }

        return err;
    }

};

#elif XPU_IS_HIP

template<typename F>
struct action_runner<constant_tag, F> {
    using data_t = typename F::data_t;
    static int call(const data_t &val) {
        return hipMemcpyToSymbol(HIP_SYMBOL(constant_memory<F>), &val, sizeof(data_t));
    }
};

template<typename K, typename... Args>
struct action_runner<kernel_tag, K, void(K::*)(kernel_context<typename K::shared_memory, typename K::constants> &, Args...)> {

    static int call(float *ms, driver_interface * /*hip_driver*/, grid g, Args... args) {
        dim block_dim = K::block_size::value;
        dim grid_dim{};

        g.get_compute_grid(block_dim, grid_dim);

        XPU_LOG("Calling kernel '%s' [block_dim = (%d, %d, %d), grid_dim = (%d, %d, %d)] with HIP driver.", type_name<K>(), block_dim.x, block_dim.y, block_dim.z, grid_dim.x, grid_dim.y, grid_dim.z);

        bool measure_time = (ms != nullptr);
        hipEvent_t start, end;
        int err = 0;

        if (measure_time) {
            SAFE_CALL(hipEventCreate(&start));
            ON_ERROR_GOTO(err, hipEventCreate(&end), cleanup_start_event);
        }

        if (measure_time) {
            ON_ERROR_GOTO(err, hipEventRecord(start), cleanup_events);
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_entry_bounded<K, K::block_size::value.linear(), Args...>), grid_dim.as_cuda_grid(), block_dim.as_cuda_grid(), 0, 0, args...);
        if (measure_time) {
            ON_ERROR_GOTO(err, hipEventRecord(end), cleanup_events);
        }
        SAFE_CALL(hipDeviceSynchronize());

        if (measure_time) {
            ON_ERROR_GOTO(err, hipEventSynchronize(end), cleanup_events);
            ON_ERROR_GOTO(err, hipEventElapsedTime(ms, start, end), cleanup_events);
            XPU_LOG("Kernel '%s' took %f ms", type_name<K>(), *ms);
        }

    cleanup_events:
        if (measure_time) {
            SAFE_CALL(hipEventDestroy(end));
        }
    cleanup_start_event:
        if (measure_time) {
            SAFE_CALL(hipEventDestroy(start));
        }

        return err;
    }

};

#endif

} // namespace xpu::detail

#undef SAFE_CALL
#undef ON_ERROR_GOTO

#endif
