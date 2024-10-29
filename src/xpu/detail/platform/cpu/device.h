#ifndef XPU_DRIVER_CPU_DEVICE_RUNTIME_H
#define XPU_DRIVER_CPU_DEVICE_RUNTIME_H

#ifndef XPU_DEVICE_H
#error "This header should not be included directly. Include xpu/device.h instead."
#endif

#include "../../macros.h"
#include "../../constant_memory.h"
#include "this_thread.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <utility>

#define XPU_DETAIL_ASSERT(x) assert(x)
#ifdef _OPENMP
#define XPU_DETAIL_OMP(...) XPU_PRAGMA(omp __VA_ARGS__)
#else
#define XPU_DETAIL_OMP(...)
#endif


// math functions
XPU_FORCE_INLINE float xpu::abs(float x) { return std::fabs(x); }
XPU_FORCE_INLINE int   xpu::abs(int a) { return std::abs(a); }
XPU_FORCE_INLINE float xpu::acos(float x) { return std::acos(x); }
XPU_FORCE_INLINE float xpu::acosh(float x) { return std::acoshf(x); }
XPU_FORCE_INLINE float xpu::acospi(float x) { return std::acos(x) / pi(); }
XPU_FORCE_INLINE float xpu::asin(float x) { return std::asin(x); }
XPU_FORCE_INLINE float xpu::asinh(float x) { return std::asinhf(x); }
XPU_FORCE_INLINE float xpu::asinpi(float x) { return asin(x) / pi(); }
XPU_FORCE_INLINE float xpu::atan2(float y, float x) { return std::atan2(y, x); }
XPU_FORCE_INLINE float xpu::atan2pi(float y, float x) { return std::atan2(y, x) / pi(); }
XPU_FORCE_INLINE float xpu::atan(float x) { return std::atan(x); }
XPU_FORCE_INLINE float xpu::atanh(float x) { return std::atanhf(x); }
XPU_FORCE_INLINE float xpu::atanpi(float x) { return std::atan(x) / pi(); }
XPU_FORCE_INLINE float xpu::cbrt(float x) { return std::cbrtf(x); }
XPU_FORCE_INLINE float xpu::ceil(float x) { return std::ceil(x); }
XPU_FORCE_INLINE float xpu::copysign(float x, float y) { return std::copysignf(x, y); }
XPU_FORCE_INLINE float xpu::cos(float x) { return std::cos(x); }
XPU_FORCE_INLINE float xpu::cosh(float x) { return std::cosh(x); }
XPU_FORCE_INLINE float xpu::cospi(float x) { return cos(x * pi()); }
XPU_FORCE_INLINE float xpu::erf(float x) { return std::erff(x); }
XPU_FORCE_INLINE float xpu::erfc(float x) { return std::erfcf(x); }
XPU_FORCE_INLINE float xpu::exp(float x) { return std::exp(x); }
XPU_FORCE_INLINE float xpu::exp2(float x) { return std::exp2f(x); }
XPU_FORCE_INLINE float xpu::exp10(float x) { return std::pow(10.0f, x); }
XPU_FORCE_INLINE float xpu::expm1(float x) { return std::expm1f(x); }
XPU_FORCE_INLINE float xpu::fdim(float x, float y) { return std::fdimf(x, y); }
XPU_FORCE_INLINE float xpu::floor(float x) { return std::floor(x); }
XPU_FORCE_INLINE float xpu::fma(float x, float y, float z) { return std::fmaf(x, y, z); }
XPU_FORCE_INLINE float xpu::fmod(float x, float y) { return std::fmod(x, y); }
// XPU_FORCE_INLINE float frexp(float x, int *nptr) { return std::frexp(x, nptr); }
XPU_FORCE_INLINE float xpu::hypot(float x, float y) { return std::hypotf(x, y); }
XPU_FORCE_INLINE int   xpu::ilogb(float x) { return std::ilogbf(x); }
XPU_FORCE_INLINE bool  xpu::isfinite(float a) { return std::isfinite(a); }
XPU_FORCE_INLINE bool  xpu::isinf(float a) { return std::isinf(a); }
XPU_FORCE_INLINE bool  xpu::isnan(float a) { return std::isnan(a); }
XPU_FORCE_INLINE float xpu::ldexp(float x, int exp) { return std::ldexp(x, exp); }
// XPU_FORCE_INLINE float lgamma(float x) { return std::lgammaf(x); }
XPU_FORCE_INLINE long long int xpu::llrint(float x) { return std::llrintf(x); }
XPU_FORCE_INLINE long long int xpu::llround(float x) { return std::llroundf(x); }
XPU_FORCE_INLINE float xpu::log(float x) { return std::log(x); }
XPU_FORCE_INLINE float xpu::log10(float x) { return std::log10(x); }
XPU_FORCE_INLINE float xpu::log1p(float x) { return std::log1pf(x); }
XPU_FORCE_INLINE float xpu::log2(float x) { return std::log2f(x); }
XPU_FORCE_INLINE float xpu::logb(float x) { return std::logbf(x); }
XPU_FORCE_INLINE long int xpu::lrint(float x) { return std::lrintf(x); }
XPU_FORCE_INLINE long int xpu::lround(float x) { return std::lroundf(x); }
XPU_FORCE_INLINE int xpu::max(int a, int b) { return std::max(a, b); }
XPU_FORCE_INLINE unsigned int xpu::max(unsigned int a, unsigned int b) { return std::max(a, b); }
XPU_FORCE_INLINE long long int xpu::max(long long int a, long long int b) { return std::max(a, b); }
XPU_FORCE_INLINE unsigned long long int xpu::max(unsigned long long int a, unsigned long long int b) { return std::max(a, b); }
XPU_FORCE_INLINE float xpu::max(float a, float b) { return std::fmaxf(a, b); }
XPU_FORCE_INLINE int xpu::min(int a, int b) { return std::min(a, b); }
XPU_FORCE_INLINE unsigned int xpu::min(unsigned int a, unsigned int b) { return std::min(a, b); }
XPU_FORCE_INLINE long long int xpu::min(long long int a, long long int b) { return std::min(a, b); }
XPU_FORCE_INLINE unsigned long long int xpu::min(unsigned long long int a, unsigned long long int b) { return std::min(a, b); }
XPU_FORCE_INLINE float xpu::min(float a, float b) { return std::fminf(a, b); }
// XPU_FORCE_INLINE float modf(float x, float *iptr) { return std::modf(x, iptr); }
XPU_FORCE_INLINE float xpu::nan(const char *tagp) { return std::nanf(tagp); }
// XPU_FORCE_INLINE float nextafter(float x, float y) { return std::nextafterf(x, y); }

inline float xpu::norm3d(float a, float b, float c) { return sqrt(a * a + b * b + c * c); }
inline float xpu::norm4d(float a, float b, float c, float d) { return sqrt(a * a + b * b + c * c + d * d); }

XPU_FORCE_INLINE float xpu::pow(float x, float y) { return std::pow(x, y); }
XPU_FORCE_INLINE float xpu::rcbrt(float x) { return 1.f / cbrt(x); }
XPU_FORCE_INLINE float xpu::remainder(float x, float y) { return std::remainderf(x, y); }
XPU_FORCE_INLINE float xpu::remquo(float x, float y, int *quo) { return std::remquof(x, y, quo); }
XPU_FORCE_INLINE float xpu::rhypot(float x, float y) { return 1.f / hypot(x, y); }
XPU_FORCE_INLINE float xpu::rint(float x) { return std::rintf(x); }
XPU_FORCE_INLINE float xpu::rnorm3d(float a, float b, float c) { return 1.f / norm3d(a, b, c); }
XPU_FORCE_INLINE float xpu::rnorm4d(float a, float b, float c, float d) { return 1.f / norm4d(a, b, c, d); }
XPU_FORCE_INLINE float xpu::round(float x) { return std::roundf(x); }
XPU_FORCE_INLINE float xpu::rsqrt(float x) { return 1.f / sqrt(x); }
XPU_FORCE_INLINE bool xpu::signbit(float a) { return std::signbit(a); }
XPU_FORCE_INLINE void xpu::sincos(float x, float *sptr, float *cptr) {
#if __APPLE__
    *sptr = std::sin(x); *cptr = std::cos(x);
#else
    ::sincosf(x, sptr, cptr);
#endif
}
XPU_FORCE_INLINE void xpu::sincospi(float x, float *sptr, float *cptr) { return sincos(x * pi(), sptr, cptr); }
XPU_FORCE_INLINE float xpu::sin(float x) { return std::sin(x); }
XPU_FORCE_INLINE float xpu::sinh(float x) { return std::sinh(x); }
XPU_FORCE_INLINE float xpu::sinpi(float x) { return sin(x * pi()); }
XPU_FORCE_INLINE float xpu::sqrt(float x) { return std::sqrt(x); }
XPU_FORCE_INLINE float xpu::tan(float x) { return std::tan(x); }
XPU_FORCE_INLINE float xpu::tanh(float x) { return std::tanh(x); }
XPU_FORCE_INLINE float xpu::tgamma(float x) { return std::tgammaf(x); }
XPU_FORCE_INLINE float xpu::trunc(float x) { return std::truncf(x); }


inline int xpu::atomic_cas(int *addr, int compare, int val) {
    __atomic_compare_exchange(addr, &compare, &val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return compare;
}

inline unsigned int xpu::atomic_cas(unsigned int *addr, unsigned int compare, unsigned int val) {
    __atomic_compare_exchange(addr, &compare, &val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return compare;
}

inline float xpu::atomic_cas(float *addr, float compare, float val) {
    __atomic_compare_exchange(addr, &compare, &val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return compare;
}

inline int xpu::atomic_cas_block(int *addr, int compare, int val) {
    return std::exchange(*addr, (*addr == compare ? val : *addr));
}

inline unsigned int xpu::atomic_cas_block(unsigned int *addr, unsigned int compare, unsigned int val) {
    return std::exchange(*addr, (*addr == compare ? val : *addr));
}

inline float xpu::atomic_cas_block(float *addr, float compare, float val) {
    return std::exchange(*addr, (*addr == compare ? val : *addr));
}

inline int xpu::atomic_add(int *addr, int val) {
    return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int xpu::atomic_add(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
}

inline float xpu::atomic_add(float *addr, float val) {
    float old = *addr;
    float assumed;

    do {
        assumed = old;
        old = atomic_cas(addr, assumed, assumed + val);
    } while (float_as_int(old) != float_as_int(assumed));

    return old;
}

inline int xpu::atomic_add_block(int *addr, int val) {
    return std::exchange(*addr, *addr + val);
}

inline unsigned int xpu::atomic_add_block(unsigned int *addr, unsigned int val) {
    return std::exchange(*addr, *addr + val);
}

inline float xpu::atomic_add_block(float *addr, float val) {
    return std::exchange(*addr, *addr + val);
}

inline int xpu::atomic_sub(int *addr, int val) {
    return __atomic_fetch_sub(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int xpu::atomic_sub(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_sub(addr, val, __ATOMIC_SEQ_CST);
}

inline int xpu::atomic_sub_block(int *addr, int val) {
    return std::exchange(*addr, *addr - val);
}

inline unsigned int xpu::atomic_sub_block(unsigned int *addr, unsigned int val) {
    return std::exchange(*addr, *addr - val);
}

inline int xpu::atomic_and(int *addr, int val) {
    return __atomic_fetch_and(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int xpu::atomic_and(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_and(addr, val, __ATOMIC_SEQ_CST);
}

inline int xpu::atomic_and_block(int *addr, int val) {
    return std::exchange(*addr, *addr & val);
}

inline unsigned int xpu::atomic_and_block(unsigned int *addr, unsigned int val) {
    return std::exchange(*addr, *addr & val);
}

inline int xpu::atomic_or(int *addr, int val) {
    return __atomic_fetch_or(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int xpu::atomic_or(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_or(addr, val, __ATOMIC_SEQ_CST);
}

inline int xpu::atomic_or_block(int *addr, int val) {
    return std::exchange(*addr, *addr | val);
}

inline unsigned int xpu::atomic_or_block(unsigned int *addr, unsigned int val) {
    return std::exchange(*addr, *addr | val);
}

inline int xpu::atomic_xor(int *addr, int val) {
    return __atomic_fetch_xor(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int xpu::atomic_xor(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_xor(addr, val, __ATOMIC_SEQ_CST);
}

inline int xpu::atomic_xor_block(int *addr, int val) {
    return std::exchange(*addr, *addr ^ val);
}

inline unsigned int xpu::atomic_xor_block(unsigned int *addr, unsigned int val) {
    return std::exchange(*addr, *addr ^ val);
}

XPU_FORCE_INLINE void xpu::barrier(xpu::tpos &) { return; }

namespace xpu {

namespace detail {
    union float_int_reint {
        float f;
        int i;
    };
} // namespace detail

inline int float_as_int(float val) {
    detail::float_int_reint xval { .f = val };
    return xval.i;
}

inline float int_as_float(int val) {
    detail::float_int_reint xval { .i = val };
    return xval.f;
}

template<typename T, int BlockSize>
class block_scan<T, BlockSize, cpu> {

public:
    struct storage_t {};

    template <typename ContextT>
    XPU_D block_scan(ContextT &, storage_t &) {}

    XPU_D block_scan(tpos &, storage_t &) {}

    XPU_D void exclusive_sum(T /*input*/, T &output) { output = T{0}; }

    template<typename ScanOp>
    XPU_D void exclusive_sum(T /*input*/, T &output, T initial_value, ScanOp /*scan_op*/) { output = initial_value; }

    XPU_D void inclusive_sum(T input, T &output) { output = input; }

    template<typename ScanOp>
    XPU_D void inclusive_sum(T input, T &output, T initial_value, ScanOp scan_op) { output = scan_op(initial_value, input); }

};

template<typename T, int BlockSize>
class block_reduce<T, BlockSize, cpu> {

public:
    struct storage_t {};

    block_reduce(tpos &, storage_t &) {}

    T sum(T input) { return input; }

    template<typename ReduceOp>
    T reduce(T input, ReduceOp /*reduce_op*/) { return input; }

};

template<typename Key, typename KeyValueType, int BlockSize, int ItemsPerThread>
class block_sort<Key, KeyValueType, BlockSize, ItemsPerThread, cpu> {

public:
    struct storage_t {};

    block_sort(tpos &, storage_t &) {}

    template<typename KeyGetter>
    KeyValueType *sort(KeyValueType *vals, size_t N, KeyValueType *, KeyGetter &&getKey) {
        std::sort(vals, &vals[N], [&](const KeyValueType &a, const KeyValueType &b) {
            return getKey(a) < getKey(b);
        });
        return vals;
    }

};

template<typename Key, int BlockSize, int ItemsPerThread>
class block_merge<Key, BlockSize, ItemsPerThread, cpu> {

public:
    struct storage_t {};

    block_merge(tpos &, storage_t &) {}

    template<typename Compare>
    void merge(const Key *a, size_t size_a, const Key *b, size_t size_b, Key *dst, Compare &&comp) {
        std::merge(a, a + size_a, b, b + size_b, dst, comp);
    }

};

} // namespace xpu

namespace xpu::detail {

template<typename F>
struct action_runner<constant_tag, F> {
    using data_t = typename F::data_t;
    static int call(const data_t &val) {
        constant_memory<F> = val;
        return 0;
    }
};

template<typename K, typename... Args>
struct action_runner<kernel_tag, K, void(K::*)(kernel_context<typename K::shared_memory, typename K::constants> &, Args...)> {

    using shared_memory = typename K::shared_memory;
    using constants = typename K::constants;
    using context = kernel_context<shared_memory, constants>;

private:
    static void kernel_step(const dim &grid_dim, const dim &block_idx, Args&&... args) {
        shared_memory smem;
        tpos pos{internal_ctor};
        constants cmem{internal_ctor};
        context ctx{internal_ctor, pos, smem, cmem};
        this_thread::grid_dim = grid_dim;
        this_thread::block_idx = block_idx;
        K{}(ctx, std::forward<Args>(args)...);
    }

public:
    static int call(kernel_launch_info launch_info, Args... args) {
        dim block_dim{1, 1, 1};
        dim grid_dim{};

        launch_info.g.get_compute_grid(block_dim, grid_dim);
        XPU_LOG("Calling kernel '%s' [block_dim = (%d, %d, %d), grid_dim = (%d, %d, %d)] with CPU driver.", type_name<K>(), block_dim.x, block_dim.y, block_dim.z, grid_dim.x, grid_dim.y, grid_dim.z);

        using clock = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<float, std::milli>;

        bool measure_time = (launch_info.ms != nullptr);
        clock::time_point start;

        if (measure_time) {
            start = clock::now();
        }

        constexpr schedule_t schedule = K::openmp::schedule;
        constexpr size_t chunk_size = K::openmp::chunk_size;

        static_assert(schedule == schedule_static || schedule == schedule_dynamic, "Unsupported OpenMP schedule");

        if constexpr (schedule == schedule_static) {

            if constexpr (chunk_size == 0) {
                XPU_DETAIL_OMP(parallel for schedule(static) collapse(3))
                for (int i = 0; i < grid_dim.x; i++) {
                    for (int j = 0; j < grid_dim.y; j++) {
                        for (int k = 0; k < grid_dim.z; k++) {
                            kernel_step(grid_dim, dim(i, j, k), std::forward<Args>(args)...);
                        }
                    }
                }
            } else {
                XPU_DETAIL_OMP(parallel for schedule(static, chunk_size) collapse(3))
                for (int i = 0; i < grid_dim.x; i++) {
                    for (int j = 0; j < grid_dim.y; j++) {
                        for (int k = 0; k < grid_dim.z; k++) {
                            kernel_step(grid_dim, dim(i, j, k), std::forward<Args>(args)...);
                        }
                    }
                }
            }
        } else if constexpr (schedule == schedule_dynamic) {
            if constexpr (chunk_size == 0) {
                XPU_DETAIL_OMP(parallel for schedule(dynamic) collapse(3))
                for (int i = 0; i < grid_dim.x; i++) {
                    for (int j = 0; j < grid_dim.y; j++) {
                        for (int k = 0; k < grid_dim.z; k++) {
                            kernel_step(grid_dim, dim(i, j, k), std::forward<Args>(args)...);
                        }
                    }
                }
            } else {
                XPU_DETAIL_OMP(parallel for schedule(dynamic, chunk_size) collapse(3))
                for (int i = 0; i < grid_dim.x; i++) {
                    for (int j = 0; j < grid_dim.y; j++) {
                        for (int k = 0; k < grid_dim.z; k++) {
                            kernel_step(grid_dim, dim(i, j, k), std::forward<Args>(args)...);
                        }
                    }
                }
            }
        } else {
            // Unreachable
        }

        if (measure_time) {
            duration elapsed = clock::now() - start;
            *launch_info.ms = elapsed.count();
            XPU_LOG("Kernel '%s' took %f ms", type_name<K>(), *launch_info.ms);
        }

        return 0;
    }

};

} // namespace xpu::detail

#endif
