
#ifndef XPU_DRIVER_CUDA_DEVICE_RUNTIME_H
#define XPU_DRIVER_CUDA_DEVICE_RUNTIME_H

#ifndef XPU_DEVICE_H
#error "This header should not be included directly. Include xpu/device.h instead."
#endif

#include "../../constant_memory.h"
#include "../../macros.h"
#include "../../parallel_merge.h"

#if XPU_IS_CUDA
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_radix_sort.cuh>
#elif XPU_IS_HIP
#include <hip/hip_runtime.h>
// #include <hipcub/hipcub.hpp> // FIXME: including hibcub main header sometimes crashes HIP clang...
#include <hipcub/block/block_scan.hpp>
#include <hipcub/block/block_reduce.hpp>
#include <hipcub/block/block_radix_sort.hpp>
#else
#error "Internal XPU error: This should never happen."
#endif

#include <iostream>
#include <type_traits>

#define XPU_DETAIL_ASSERT(x) static_cast<void>(0)


XPU_D XPU_FORCE_INLINE int xpu::abs(int a) { return ::abs(a); }
XPU_D XPU_FORCE_INLINE float xpu::abs(float x) { return ::fabsf(x); }
XPU_D XPU_FORCE_INLINE float xpu::acos(float x) { return ::acosf(x); }
XPU_D XPU_FORCE_INLINE float xpu::acosh(float x) { return ::acoshf(x); }
XPU_D XPU_FORCE_INLINE float xpu::acospi(float x) { return ::acosf(x) / pi(); }
XPU_D XPU_FORCE_INLINE float xpu::asin(float x) { return ::asinf(x); }
XPU_D XPU_FORCE_INLINE float xpu::asinh(float x) { return ::asinhf(x); }
XPU_D XPU_FORCE_INLINE float xpu::asinpi(float x) { return ::asinf(x) / pi(); }
XPU_D XPU_FORCE_INLINE float xpu::atan2(float y, float x) { return ::atan2f(y, x); }
XPU_D XPU_FORCE_INLINE float xpu::atan2pi(float y, float x) { return ::atan2f(y, x) / pi(); }
XPU_D XPU_FORCE_INLINE float xpu::atan(float x) { return ::atanf(x); }
XPU_D XPU_FORCE_INLINE float xpu::atanh(float x) { return ::atanhf(x); }
XPU_D XPU_FORCE_INLINE float xpu::atanpi(float x) { return ::atanf(x) / pi(); }
XPU_D XPU_FORCE_INLINE float xpu::cbrt(float x) { return ::cbrtf(x); }
XPU_D XPU_FORCE_INLINE float xpu::ceil(float x) { return ::ceilf(x); }
XPU_D XPU_FORCE_INLINE float xpu::copysign(float x, float y) { return ::copysignf(x, y); }
XPU_D XPU_FORCE_INLINE float xpu::cos(float x) { return ::cosf(x); }
XPU_D XPU_FORCE_INLINE float xpu::cosh(float x) { return ::coshf(x); }
XPU_D XPU_FORCE_INLINE float xpu::cospi(float x) { return ::cospi(x); }
XPU_D XPU_FORCE_INLINE float xpu::erf(float x) { return ::erff(x); }
XPU_D XPU_FORCE_INLINE float xpu::erfc(float x) { return ::erfcf(x); }
XPU_D XPU_FORCE_INLINE float xpu::exp(float x) { return ::expf(x); }
XPU_D XPU_FORCE_INLINE float xpu::exp2(float x) { return ::exp2f(x); }
XPU_D XPU_FORCE_INLINE float xpu::exp10(float x) { return ::exp10f(x); }
XPU_D XPU_FORCE_INLINE float xpu::expm1(float x) { return ::expm1(x); }
XPU_D XPU_FORCE_INLINE float xpu::fdim(float x, float y) { return ::fdimf(x, y); }
XPU_D XPU_FORCE_INLINE float xpu::floor(float x) { return ::floorf(x); }
XPU_D XPU_FORCE_INLINE float xpu::fma(float x, float y, float z) { return ::fmaf(x, y, z); }
XPU_D XPU_FORCE_INLINE float xpu::fmod(float x, float y) { return ::fmodf(x, y); }
XPU_D XPU_FORCE_INLINE float xpu::hypot(float x, float y) { return ::hypotf(x, y); }
XPU_D XPU_FORCE_INLINE int xpu::ilogb(float x) { return ::ilogbf(x); }
XPU_D XPU_FORCE_INLINE bool xpu::isfinite(float a) { return ::isfinite(a); }
XPU_D XPU_FORCE_INLINE bool xpu::isinf(float x) { return ::isinf(x); }
XPU_D XPU_FORCE_INLINE bool xpu::isnan(float x) { return ::isnan(x); }
XPU_D XPU_FORCE_INLINE float xpu::ldexp(float x, int exp) { return ::ldexpf(x, exp); }
XPU_D XPU_FORCE_INLINE long long int xpu::llrint(float x) { return ::llrintf(x); }
XPU_D XPU_FORCE_INLINE long long int xpu::llround(float x) { return ::llroundf(x); }
XPU_D XPU_FORCE_INLINE float xpu::log(float x) { return ::logf(x); }
XPU_D XPU_FORCE_INLINE float xpu::log10(float x) { return ::log10f(x); }
XPU_D XPU_FORCE_INLINE float xpu::log1p(float x) { return ::log1pf(x); }
XPU_D XPU_FORCE_INLINE float xpu::log2(float x) { return ::log2f(x); }
XPU_D XPU_FORCE_INLINE float xpu::logb(float x) { return ::logbf(x); }
XPU_D XPU_FORCE_INLINE long int xpu::lrint(float x) { return ::lrintf(x); }
XPU_D XPU_FORCE_INLINE long int xpu::lround(float x) { return ::lroundf(x); }
XPU_D XPU_FORCE_INLINE int xpu::max(int a, int b) { return ::max(a, b); }
XPU_D XPU_FORCE_INLINE unsigned int xpu::max(unsigned int a, unsigned int b) { return ::max(a, b); }
XPU_D XPU_FORCE_INLINE long long int xpu::max(long long int a, long long int b) { return ::max(a, b); }
XPU_D XPU_FORCE_INLINE unsigned long long int xpu::max(unsigned long long int a, unsigned long long int b) { return ::max(a, b); }
XPU_D XPU_FORCE_INLINE float xpu::max(float a, float b) { return ::fmaxf(a, b); }
XPU_D XPU_FORCE_INLINE int xpu::min(int a, int b) { return ::min(a, b); }
XPU_D XPU_FORCE_INLINE unsigned int xpu::min(unsigned int a, unsigned int b) { return ::min(a, b); }
XPU_D XPU_FORCE_INLINE long long int xpu::min(long long int a, long long int b) { return ::min(a, b); }
XPU_D XPU_FORCE_INLINE unsigned long long int xpu::min(unsigned long long int a, unsigned long long int b) { return ::min(a, b); }
XPU_D XPU_FORCE_INLINE float xpu::min(float a, float b) { return ::fminf(a, b); }
XPU_D XPU_FORCE_INLINE float xpu::nan(const char *tagp) { return ::nanf(tagp); }
XPU_D XPU_FORCE_INLINE float xpu::norm3d(float a, float b, float c) { return ::norm3df(a, b, c); }
XPU_D XPU_FORCE_INLINE float xpu::norm4d(float a, float b, float c, float d) { return ::norm4df(a, b, c, d); }
// XPU_D XPU_FORCE_INLINE float normcdf(float y) { return ::normcdff(y); }
// XPU_D XPU_FORCE_INLINE float normcdfinv(float y) { return ::normcdfinvf(y); }
XPU_D XPU_FORCE_INLINE float xpu::pow(float x, float y) { return ::powf(x, y); }
XPU_D XPU_FORCE_INLINE float xpu::rcbrt(float x) { return ::rcbrtf(x); }
XPU_D XPU_FORCE_INLINE float xpu::remainder(float x, float y) { return ::remainderf(x, y); }
XPU_D XPU_FORCE_INLINE float xpu::remquo(float x, float y, int *quo) { return ::remquof(x, y, quo); }
XPU_D XPU_FORCE_INLINE float xpu::rhypot(float x, float y) { return ::rhypotf(x, y); }
XPU_D XPU_FORCE_INLINE float xpu::rint(float x) { return ::rintf(x); }
XPU_D XPU_FORCE_INLINE float xpu::rnorm3d(float a, float b, float c) { return ::rnorm3df(a, b, c); }
XPU_D XPU_FORCE_INLINE float xpu::rnorm4d(float a, float b, float c, float d) { return ::rnorm4df(a, b, c, d); }
XPU_D XPU_FORCE_INLINE float xpu::round(float x) { return ::roundf(x); }
XPU_D XPU_FORCE_INLINE float xpu::rsqrt(float x) { return ::rsqrtf(x); }
XPU_D XPU_FORCE_INLINE bool xpu::signbit(float a) { return ::signbit(a); }
XPU_D XPU_FORCE_INLINE void xpu::sincos(float x, float *sptr, float *cptr) { return ::sincosf(x, sptr, cptr); }
XPU_D XPU_FORCE_INLINE void xpu::sincospi(float x, float *sptr, float *cptr) { return ::sincospif(x, sptr, cptr); }
XPU_D XPU_FORCE_INLINE float xpu::sin(float x) { return ::sinf(x); }
XPU_D XPU_FORCE_INLINE float xpu::sinh(float x) { return ::sinhf(x); }
XPU_D XPU_FORCE_INLINE float xpu::sinpi(float x) { return ::sinpif(x); }
XPU_D XPU_FORCE_INLINE float xpu::sqrt(float x) { return ::sqrtf(x); }
XPU_D XPU_FORCE_INLINE float xpu::tan(float x) { return ::tanf(x); }
XPU_D XPU_FORCE_INLINE float xpu::tanh(float x) { return ::tanhf(x); }
XPU_D XPU_FORCE_INLINE float xpu::tgamma(float x) { return ::tgammaf(x); }
XPU_D XPU_FORCE_INLINE float xpu::trunc(float x) { return ::truncf(x); }


XPU_D XPU_FORCE_INLINE int xpu::atomic_cas(int *addr, int compare, int val) {
    return atomicCAS(addr, compare, val);
}

XPU_D XPU_FORCE_INLINE unsigned int xpu::atomic_cas(unsigned int *addr, unsigned int compare, unsigned int val) {
    return atomicCAS(addr, compare, val);
}

XPU_D XPU_FORCE_INLINE float xpu::atomic_cas(float *addr, float compare, float val) {
    return __int_as_float(atomicCAS((int *) addr, __float_as_int(compare), __float_as_int(val)));
}

XPU_D XPU_FORCE_INLINE int xpu::atomic_cas_block(int *addr, int compare, int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicCAS_block(addr, compare, val);
#else
    return atomicCAS(addr, compare, val);
#endif
}

XPU_D XPU_FORCE_INLINE unsigned int xpu::atomic_cas_block(unsigned int *addr, unsigned int compare, unsigned int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicCAS_block(addr, compare, val);
#else
    return atomicCAS(addr, compare, val);
#endif
}

XPU_D XPU_FORCE_INLINE float xpu::atomic_cas_block(float *addr, float compare, float val) {
    return int_as_float(atomic_cas_block((int *) addr, float_as_int(compare), float_as_int(val)));
}

XPU_D XPU_FORCE_INLINE int xpu::atomic_add(int *addr, int val) {
    return atomicAdd(addr, val);
}

XPU_D XPU_FORCE_INLINE unsigned int xpu::atomic_add(unsigned int *addr, unsigned int val) {
    return atomicAdd(addr, val);
}

XPU_D XPU_FORCE_INLINE float xpu::atomic_add(float *addr, float val) {
    return atomicAdd(addr, val);
}

XPU_D XPU_FORCE_INLINE int xpu::atomic_add_block(int *addr, int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicAdd_block(addr, val);
#else
    return atomicAdd(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE unsigned int xpu::atomic_add_block(unsigned int *addr, unsigned int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicAdd_block(addr, val);
#else
    return atomicAdd(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE float xpu::atomic_add_block(float *addr, float val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicAdd_block(addr, val);
#else
    return atomicAdd(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE int xpu::atomic_sub(int *addr, int val) {
    return atomicSub(addr, val);
}

XPU_D XPU_FORCE_INLINE unsigned int xpu::atomic_sub(unsigned int *addr, unsigned int val) {
    return atomicSub(addr, val);
}

XPU_D XPU_FORCE_INLINE int xpu::atomic_sub_block(int *addr, int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicSub_block(addr, val);
#else
    return atomicSub(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE unsigned int xpu::atomic_sub_block(unsigned int *addr, unsigned int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicSub_block(addr, val);
#else
    return atomicSub(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE int xpu::atomic_and(int *addr, int val) {
    return atomicAnd(addr, val);
}

XPU_D XPU_FORCE_INLINE unsigned int xpu::atomic_and(unsigned int *addr, unsigned int val) {
    return atomicAnd(addr, val);
}

XPU_D XPU_FORCE_INLINE int xpu::atomic_and_block(int *addr, int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicAnd_block(addr, val);
#else
    return atomicAnd(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE unsigned int xpu::atomic_and_block(unsigned int *addr, unsigned int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicAnd_block(addr, val);
#else
    return atomicAnd(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE int xpu::atomic_or(int *addr, int val) {
    return atomicOr(addr, val);
}

XPU_D XPU_FORCE_INLINE unsigned int xpu::atomic_or(unsigned int *addr, unsigned int val) {
    return atomicOr(addr, val);
}

XPU_D XPU_FORCE_INLINE int xpu::atomic_or_block(int *addr, int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicOr_block(addr, val);
#else
    return atomicOr(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE unsigned int xpu::atomic_or_block(unsigned int *addr, unsigned int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicOr_block(addr, val);
#else
    return atomicOr(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE int xpu::atomic_xor(int *addr, int val) {
    return atomicXor(addr, val);
}

XPU_D XPU_FORCE_INLINE unsigned int xpu::atomic_xor(unsigned int *addr, unsigned int val) {
    return atomicXor(addr, val);
}

XPU_D XPU_FORCE_INLINE int xpu::atomic_xor_block(int *addr, int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicXor_block(addr, val);
#else
    return atomicXor(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE unsigned int xpu::atomic_xor_block(unsigned int *addr, unsigned int val) {
#if XPU_CUDA_HAS_BLOCK_ATOMICS
    return atomicXor_block(addr, val);
#else
    return atomicXor(addr, val);
#endif
}

XPU_D XPU_FORCE_INLINE void xpu::barrier(tpos &) { __syncthreads(); }

XPU_D XPU_FORCE_INLINE int xpu::float_as_int(float val) { return __float_as_int(val); }
XPU_D XPU_FORCE_INLINE float xpu::int_as_float(int val) { return __int_as_float(val); }

namespace xpu {

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

    template <typename ContextT>
    XPU_D block_scan(ContextT &, storage_t &st) : impl(st.scanTemp) {}

    XPU_D block_scan(tpos &, storage_t &st) : impl(st.scanTemp) {}

    XPU_D void exclusive_sum(T input, T &output) { impl.ExclusiveSum(input, output); }

    template<typename ScanOp>
    XPU_D void exclusive_sum(T input, T &output, T initial_value, ScanOp scan_op) {
        impl.ExclusiveSum(input, output, initial_value, scan_op);
    }

    XPU_D void inclusive_sum(T input, T &output) { impl.InclusiveSum(input, output); }

    template<typename ScanOp>
    XPU_D void inclusive_sum(T input, T &output, T initial_value, ScanOp scan_op) {
        impl.InclusiveSum(input, output, initial_value, scan_op);
    }

private:
    block_scan_impl impl;

};

template<typename T, int BlockSize>
class block_reduce<T, BlockSize, XPU_COMPILATION_TARGET> {

private:
    using impl_t = detail::cub::BlockReduce<T, BlockSize>;

public:
    using storage_t = typename impl_t::TempStorage;

    template <typename ContextT>
    XPU_D block_reduce(ContextT &, storage_t &st) : impl(st) {}

    XPU_D block_reduce(tpos &, storage_t &st) : impl(st) {}

    /**
     * @brief Compute the sum of all values in the block.
     *
     * @param input The value to be summed.
     * @return The sum of all values in the block to thread 0. Return value to other threads is undefined.
     */
    XPU_D T sum(T input) { return impl.Sum(input); }

    /**
     * @brief Compute reduction of all values in the block.
     *
     * @param input The value to be reduced.
     * @param reduce_op The reduction operation.
     * @return The reduction of all values in the block to thread 0. Return value to other threads is undefined.
     */
    template<typename ReduceOp>
    XPU_D T reduce(T input, ReduceOp reduce_op) { return impl.Reduce(input, reduce_op); }

private:
    impl_t impl;

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
    using impl_t = detail::parallel_merge<Key, BlockSize, ItemsPerThread>;

    static_assert(std::is_trivially_constructible_v<data_t>, "Merged type needs trivial constructor.");

    using storage_t = typename impl_t::storage_t;

    XPU_D block_merge(tpos& pos, storage_t &storage) : impl(pos, storage) {}

    template<typename Compare>
    XPU_D void merge(const data_t *a, size_t size_a, const data_t *b, size_t size_b, data_t *dst, Compare &&comp) {
        impl.merge(a, size_a, b, size_b, dst, comp);
    }

private:
    impl_t impl;

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

    static int call(kernel_launch_info launch_info, Args... args) {
        dim block_dim = K::block_size::value;
        dim grid_dim{};

        launch_info.g.get_compute_grid(block_dim, grid_dim);

        XPU_LOG("Calling kernel '%s' [block_dim = (%d, %d, %d), grid_dim = (%d, %d, %d)] with CUDA driver.", type_name<K>(), block_dim.x, block_dim.y, block_dim.z, grid_dim.x, grid_dim.y, grid_dim.z);

        bool measure_time = (launch_info.ms != nullptr);
        cudaEvent_t start, end;
        int err = 0;

        if (measure_time) {
            SAFE_CALL(cudaEventCreate(&start));
            ON_ERROR_GOTO(err, cudaEventCreate(&end), cleanup_start_event);
        }

        if (measure_time) {
            ON_ERROR_GOTO(err, cudaEventRecord(start), cleanup_events);
        }

        if (launch_info.queue_handle == nullptr) {
            kernel_entry_bounded<K, K::block_size::value.linear(), Args...><<<grid_dim.as_cuda_grid(), block_dim.as_cuda_grid()>>>(args...);
        } else {
            cudaStream_t stream = static_cast<cudaStream_t>(launch_info.queue_handle);
            kernel_entry_bounded<K, K::block_size::value.linear(), Args...><<<grid_dim.as_cuda_grid(), block_dim.as_cuda_grid(), 0, stream>>>(args...);
        }


        if (measure_time) {
            ON_ERROR_GOTO(err, cudaEventRecord(end), cleanup_events);
        }
        SAFE_CALL(cudaDeviceSynchronize());

        if (measure_time) {
            ON_ERROR_GOTO(err, cudaEventSynchronize(end), cleanup_events);
            float ms;
            ON_ERROR_GOTO(err, cudaEventElapsedTime(&ms, start, end), cleanup_events);
            *launch_info.ms = ms;
            XPU_LOG("Kernel '%s' took %f ms", type_name<K>(), *launch_info.ms);
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

    static int call(kernel_launch_info launch_info, Args... args) {
        dim block_dim = K::block_size::value;
        dim grid_dim{};

        launch_info.g.get_compute_grid(block_dim, grid_dim);

        XPU_LOG("Calling kernel '%s' [block_dim = (%d, %d, %d), grid_dim = (%d, %d, %d)] with HIP driver.", type_name<K>(), block_dim.x, block_dim.y, block_dim.z, grid_dim.x, grid_dim.y, grid_dim.z);

        bool measure_time = (launch_info.ms != nullptr);
        hipStream_t stream = static_cast<hipStream_t>(launch_info.queue_handle);
        hipEvent_t start, end;
        int err = 0;

        if (measure_time) {
            SAFE_CALL(hipEventCreate(&start));
            ON_ERROR_GOTO(err, hipEventCreate(&end), cleanup_start_event);
        }

        if (measure_time) {
            ON_ERROR_GOTO(err, hipEventRecord(start), cleanup_events);
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_entry_bounded<K, K::block_size::value.linear(), Args...>), grid_dim.as_cuda_grid(), block_dim.as_cuda_grid(), 0, stream, std::forward<Args>(args)...);
        if (measure_time) {
            ON_ERROR_GOTO(err, hipEventRecord(end), cleanup_events);
        }
        SAFE_CALL(hipDeviceSynchronize());

        if (measure_time) {
            ON_ERROR_GOTO(err, hipEventSynchronize(end), cleanup_events);
            float ms;
            ON_ERROR_GOTO(err, hipEventElapsedTime(&ms, start, end), cleanup_events);
            *launch_info.ms = ms;
            XPU_LOG("Kernel '%s' took %f ms", type_name<K>(), *launch_info.ms);
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
