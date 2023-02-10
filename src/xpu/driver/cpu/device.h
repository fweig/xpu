#ifndef XPU_DRIVER_CPU_DEVICE_RUNTIME_H
#define XPU_DRIVER_CPU_DEVICE_RUNTIME_H

#ifndef XPU_DEVICE_H
#error "This header should not be included directly. Include xpu/device.h instead."
#endif

#include "../../detail/macros.h"
#include "../../detail/constant_memory.h"
#include "this_thread.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

#define XPU_DETAIL_ASSERT(x) assert(x)

namespace xpu {

template<>
class tpos_impl<driver_t::cpu> {

public:
    inline int thread_idx_x() const { return 0; }
    inline int thread_idx_y() const { return 0; }
    inline int thread_idx_z() const { return 0; }

    inline int block_dim_x() const { return 1; }
    inline int block_dim_y() const { return 1; }
    inline int block_dim_z() const { return 1; }

    inline int block_idx_x() const { return detail::this_thread::block_idx.x; }
    inline int block_idx_y() const { return detail::this_thread::block_idx.y; }
    inline int block_idx_z() const { return detail::this_thread::block_idx.z; }

    inline int grid_dim_x() const { return detail::this_thread::grid_dim.x; }
    inline int grid_dim_y() const { return detail::this_thread::grid_dim.y; }
    inline int grid_dim_z() const { return detail::this_thread::grid_dim.z; }
};

namespace detail {

template<typename C>
class cmem_impl_leaf {
protected:
    XPU_D const typename C::data_t &access() const { return detail::constant_memory<C>; }
};

template<typename...>
class cmem_impl_base {};

template<typename C, typename... ConstantsTail>
class cmem_impl_base<C, ConstantsTail...> : public cmem_impl_leaf<C>, public cmem_impl_base<ConstantsTail...> {
};

} // namespace detail

template<typename... Constants>
class cmem_impl<driver_t::cpu, Constants...> : public detail::cmem_impl_base<Constants...> {
public:
    template<typename Constant>
    XPU_D std::enable_if_t<(std::is_same_v<Constant, Constants> || ...), const typename Constant::data_t &>  get() const {
        return detail::cmem_impl_leaf<Constant>::access();
    }
};

// math functions
XPU_FORCE_INLINE float abs(float x) { return std::fabs(x); }
XPU_FORCE_INLINE int   abs(int a) { return std::abs(a); }
XPU_FORCE_INLINE float acos(float x) { return std::acos(x); }
XPU_FORCE_INLINE float acosh(float x) { return std::acoshf(x); }
XPU_FORCE_INLINE float asin(float x) { return std::asin(x); }
XPU_FORCE_INLINE float asinh(float x) { return std::asinhf(x); }
XPU_FORCE_INLINE float atan2(float y, float x) { return std::atan2(y, x); }
XPU_FORCE_INLINE float atan(float x) { return std::atan(x); }
XPU_FORCE_INLINE float atanh(float x) { return std::atanhf(x); }
XPU_FORCE_INLINE float cbrt(float x) { return std::cbrtf(x); }
XPU_FORCE_INLINE float ceil(float x) { return std::ceil(x); }
XPU_FORCE_INLINE float copysign(float x, float y) { return std::copysignf(x, y); }
XPU_FORCE_INLINE float cos(float x) { return std::cos(x); }
XPU_FORCE_INLINE float cosh(float x) { return std::cosh(x); }
XPU_FORCE_INLINE float cospi(float x) { return cos(x * pi()); }
XPU_FORCE_INLINE float erf(float x) { return std::erff(x); }
XPU_FORCE_INLINE float erfc(float x) { return std::erfcf(x); }
XPU_FORCE_INLINE float exp2(float x) { return std::exp2f(x); }
XPU_FORCE_INLINE float exp(float x) { return std::exp(x); }
XPU_FORCE_INLINE float expm1(float x) { return std::expm1f(x); }
XPU_FORCE_INLINE float fdim(float x, float y) { return std::fdimf(x, y); }
XPU_FORCE_INLINE float floor(float x) { return std::floor(x); }
XPU_FORCE_INLINE float fma(float x, float y, float z) { return std::fmaf(x, y, z); }
XPU_FORCE_INLINE float fmod(float x, float y) { return std::fmod(x, y); }
// XPU_FORCE_INLINE float frexp(float x, int *nptr) { return std::frexp(x, nptr); }
XPU_FORCE_INLINE float hypot(float x, float y) { return std::hypotf(x, y); }
XPU_FORCE_INLINE int ilogb(float x) { return std::ilogbf(x); }
XPU_FORCE_INLINE bool isfinite(float a) { return std::isfinite(a); }
XPU_FORCE_INLINE bool isinf(float a) { return std::isinf(a); }
XPU_FORCE_INLINE bool isnan(float a) { return std::isnan(a); }
XPU_FORCE_INLINE float j0(float x) {
#if __APPLE__
    return ::j0(x);
#else
    return ::j0f(x);
#endif
}
XPU_FORCE_INLINE float j1(float x) {
#if __APPLE__
    return ::j1(x);
#else
    return ::j1f(x);
#endif
}
XPU_FORCE_INLINE float jn(int n, float x) {
#if __APPLE__
    return ::jn(n, x);
#else
    return ::jnf(n, x);
#endif
}
XPU_FORCE_INLINE float ldexp(float x, int exp) { return std::ldexp(x, exp); }
// XPU_FORCE_INLINE float lgamma(float x) { return std::lgammaf(x); }
XPU_FORCE_INLINE long long int llrint(float x) { return std::llrintf(x); }
XPU_FORCE_INLINE long long int llround(float x) { return std::llroundf(x); }
XPU_FORCE_INLINE float log(float x) { return std::log(x); }
XPU_FORCE_INLINE float log10(float x) { return std::log10(x); }
XPU_FORCE_INLINE float log1p(float x) { return std::log1pf(x); }
XPU_FORCE_INLINE float log2(float x) { return std::log2f(x); }
XPU_FORCE_INLINE float logb(float x) { return std::logbf(x); }
XPU_FORCE_INLINE long int lrint(float x) { return std::lrintf(x); }
XPU_FORCE_INLINE long int lround(float x) { return std::lroundf(x); }
XPU_FORCE_INLINE int max(int a, int b) { return std::max(a, b); }
XPU_FORCE_INLINE unsigned int max(unsigned int a, unsigned int b) { return std::max(a, b); }
XPU_FORCE_INLINE long long int max(long long int a, long long int b) { return std::max(a, b); }
XPU_FORCE_INLINE unsigned long long int max(unsigned long long int a, unsigned long long int b) { return std::max(a, b); }
XPU_FORCE_INLINE float max(float a, float b) { return std::fmaxf(a, b); }
XPU_FORCE_INLINE int min(int a, int b) { return std::min(a, b); }
XPU_FORCE_INLINE unsigned int min(unsigned int a, unsigned int b) { return std::min(a, b); }
XPU_FORCE_INLINE long long int min(long long int a, long long int b) { return std::min(a, b); }
XPU_FORCE_INLINE unsigned long long int min(unsigned long long int a, unsigned long long int b) { return std::min(a, b); }
XPU_FORCE_INLINE float min(float a, float b) { return std::fminf(a, b); }
// XPU_FORCE_INLINE float modf(float x, float *iptr) { return std::modf(x, iptr); }
XPU_FORCE_INLINE float nan(const char *tagp) { return std::nanf(tagp); }
XPU_FORCE_INLINE float nearbyint(float x) { return std::nearbyintf(x); }
// XPU_FORCE_INLINE float nextafter(float x, float y) { return std::nextafterf(x, y); }

inline float norm(int dim, const float *a) {
    float aggr = 0;
    for (int i = 0; i < dim; i++) {
        aggr += a[i] * a[i];
    }
    return sqrt(aggr);
}

inline float norm3d(float a, float b, float c) {
    return sqrt(a * a + b * b + c * c);
}

inline float norm4d(float a, float b, float c, float d) {
    return sqrt(a * a + b * b + c * c + d * d);
}

XPU_FORCE_INLINE float pow(float x, float y) { return std::pow(x, y); }
XPU_FORCE_INLINE float rcbrt(float x) { return 1.f / cbrt(x); }
XPU_FORCE_INLINE float remainder(float x, float y) { return std::remainderf(x, y); }
XPU_FORCE_INLINE float remquo(float x, float y, int *quo) { return std::remquof(x, y, quo); }
XPU_FORCE_INLINE float rhypot(float x, float y) { return 1.f / hypot(x, y); }
XPU_FORCE_INLINE float rint(float x) { return std::rintf(x); }
XPU_FORCE_INLINE float rnorm(int dim, const float *a) { return 1.f / norm(dim, a); }
XPU_FORCE_INLINE float rnorm3d(float a, float b, float c) { return 1.f / norm3d(a, b, c); }
XPU_FORCE_INLINE float rnorm4d(float a, float b, float c, float d) { return 1.f / norm4d(a, b, c, d); }
XPU_FORCE_INLINE float round(float x) { return std::roundf(x); }
XPU_FORCE_INLINE float rsqrt(float x) { return 1.f / sqrt(x); }
XPU_FORCE_INLINE float scalbln(float x, long int n) { return std::scalblnf(x, n); }
XPU_FORCE_INLINE float scalbn(float x, int n) { return std::scalbnf(x, n); }
XPU_FORCE_INLINE bool signbit(float a) { return std::signbit(a); }
XPU_FORCE_INLINE void sincos(float x, float *sptr, float *cptr) {
#if __APPLE__
    *sptr = std::sin(x); *cptr = std::cos(x);
#else
    return ::sincosf(x, sptr, cptr);
#endif
}
XPU_FORCE_INLINE void sincospi(float x, float *sptr, float *cptr) { return sincos(x * pi(), sptr, cptr); }
XPU_FORCE_INLINE float sin(float x) { return std::sin(x); }
XPU_FORCE_INLINE float sinh(float x) { return std::sinh(x); }
XPU_FORCE_INLINE float sinpi(float x) { return sin(x * pi()); }
XPU_FORCE_INLINE float sqrt(float x) { return std::sqrt(x); }
XPU_FORCE_INLINE float tan(float x) { return std::tan(x); }
XPU_FORCE_INLINE float tanh(float x) { return std::tanh(x); }
XPU_FORCE_INLINE float tgamma(float x) { return std::tgammaf(x); }
XPU_FORCE_INLINE float trunc(float x) { return std::truncf(x); }
XPU_FORCE_INLINE float y0(float x) {
#if __APPLE__
    return ::y0(x);
#else
    return ::y0f(x);
#endif
}
XPU_FORCE_INLINE float y1(float x) {
#if __APPLE__
    return ::y1(x);
#else
    return ::y1f(x);
#endif
}
XPU_FORCE_INLINE float yn(int n, float x) {
#if __APPLE__
    return ::yn(n, x);
#else
    return ::ynf(n, x);
#endif
}

inline int atomic_cas(int *addr, int compare, int val) {
    __atomic_compare_exchange(addr, &compare, &val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return compare;
}

inline unsigned int atomic_cas(unsigned int *addr, unsigned int compare, unsigned int val) {
    __atomic_compare_exchange(addr, &compare, &val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return compare;
}

inline float atomic_cas(float *addr, float compare, float val) {
    __atomic_compare_exchange(addr, &compare, &val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return compare;
}

inline int atomic_cas_block(int *addr, int compare, int val) {
    return std::exchange(*addr, (*addr == compare ? val : *addr));
}

inline unsigned int atomic_cas_block(unsigned int *addr, unsigned int compare, unsigned int val) {
    return std::exchange(*addr, (*addr == compare ? val : *addr));
}

inline float atomic_cas_block(float *addr, float compare, float val) {
    return std::exchange(*addr, (*addr == compare ? val : *addr));
}

inline int atomic_add(int *addr, int val) {
    return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int atomic_add(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
}

inline float atomic_add(float *addr, float val) {
    float old = *addr;
    float assumed;

    do {
        assumed = old;
        old = atomic_cas(addr, assumed, assumed + val);
    } while (float_as_int(old) != float_as_int(assumed));

    return old;
}

inline int atomic_add_block(int *addr, int val) {
    return std::exchange(*addr, *addr + val);
}

inline unsigned int atomic_add_block(unsigned int *addr, unsigned int val) {
    return std::exchange(*addr, *addr + val);
}

inline float atomic_add_block(float *addr, float val) {
    return std::exchange(*addr, *addr + val);
}

inline int atomic_sub(int *addr, int val) {
    return __atomic_fetch_sub(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int atomic_sub(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_sub(addr, val, __ATOMIC_SEQ_CST);
}

inline int atomic_sub_block(int *addr, int val) {
    return std::exchange(*addr, *addr - val);
}

inline unsigned int atomic_sub_block(unsigned int *addr, unsigned int val) {
    return std::exchange(*addr, *addr - val);
}

inline int atomic_and(int *addr, int val) {
    return __atomic_fetch_and(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int atomic_and(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_and(addr, val, __ATOMIC_SEQ_CST);
}

inline int atomic_and_block(int *addr, int val) {
    return std::exchange(*addr, *addr & val);
}

inline unsigned int atomic_and_block(unsigned int *addr, unsigned int val) {
    return std::exchange(*addr, *addr & val);
}

inline int atomic_or(int *addr, int val) {
    return __atomic_fetch_or(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int atomic_or(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_or(addr, val, __ATOMIC_SEQ_CST);
}

inline int atomic_or_block(int *addr, int val) {
    return std::exchange(*addr, *addr | val);
}

inline unsigned int atomic_or_block(unsigned int *addr, unsigned int val) {
    return std::exchange(*addr, *addr | val);
}

inline int atomic_xor(int *addr, int val) {
    return __atomic_fetch_xor(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int atomic_xor(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_xor(addr, val, __ATOMIC_SEQ_CST);
}

inline int atomic_xor_block(int *addr, int val) {
    return std::exchange(*addr, *addr ^ val);
}

inline unsigned int atomic_xor_block(unsigned int *addr, unsigned int val) {
    return std::exchange(*addr, *addr ^ val);
}

XPU_FORCE_INLINE void barrier() { return; }

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

    XPU_D block_scan(storage_t &) {}

    XPU_D void exclusive_sum(T /*input*/, T &output) { output = T{0}; }

    template<typename ScanOp>
    XPU_D void exclusive_sum(T /*input*/, T &output, T initial_value, ScanOp /*scan_op*/) { output = initial_value; }

    template<int ItemsPerThread>
    XPU_D void exclusive_sum(T(&input)[ItemsPerThread], T(&output)[ItemsPerThread]) {
        exclusive_sum(input, output, T{0}, [](T a, T b) { return a + b; });
    }

    template<int ItemsPerThread, typename ScanOp>
    XPU_D void exclusive_sum(T(&input)[ItemsPerThread], T(&output)[ItemsPerThread], T initial_value, ScanOp scan_op) {
        static_assert(ItemsPerThread > 0);
        output[0] = initial_value;
        for (int i = 1; i < ItemsPerThread; i++) {
            output[i] = scan_op(output[i - 1], input[i]);
        }
    }

    XPU_D void inclusive_sum(T input, T &output) { output = input; }

    template<typename ScanOp>
    XPU_D void inclusive_sum(T input, T &output, T initial_value, ScanOp scan_op) { output = scan_op(initial_value, input); }

    template<int ItemsPerThread>
    XPU_D void inclusive_sum(T(&input)[ItemsPerThread], T(&output)[ItemsPerThread]) {
        inclusive_sum(input, output, T{0}, [](T a, T b) { return a + b; });
    }

    template<int ItemsPerThread, typename ScanOp>
    XPU_D void inclusive_sum(T(&input)[ItemsPerThread], T(&output)[ItemsPerThread], T initial_value, ScanOp scan_op) {
        static_assert(ItemsPerThread > 0);
        output[0] = scan_op(initial_value, input[0]);
        for (int i = 1; i < ItemsPerThread; i++) {
            output[i] = scan_op(output[i - 1], input[i]);
        }
    }
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
    using context = kernel_context<shared_memory>;

    static int call(float *ms, driver_interface * /*cpu_driver*/, grid g, Args... args) {
        dim block_dim{1, 1, 1};
        dim grid_dim{};

        g.get_compute_grid(block_dim, grid_dim);
        XPU_LOG("Calling kernel '%s' [block_dim = (%d, %d, %d), grid_dim = (%d, %d, %d)] with CPU driver.", type_name<K>(), block_dim.x, block_dim.y, block_dim.z, grid_dim.x, grid_dim.y, grid_dim.z);

        using clock = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<float, std::milli>;

        bool measure_time = (ms != nullptr);
        clock::time_point start;

        if (measure_time) {
            start = clock::now();
        }

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) collapse(3)
        #endif
        for (int i = 0; i < grid_dim.x; i++) {
            for (int j = 0; j < grid_dim.y; j++) {
                for (int k = 0; k < grid_dim.z; k++) {
                    shared_memory smem;
                    tpos pos{};
                    constants cmem{};
                    kernel_context ctx{pos, smem, cmem};
                    this_thread::block_idx = dim{i, j, k};
                    this_thread::grid_dim = grid_dim;
                    K{}(ctx, args...);
                }
            }
        }

        if (measure_time) {
            duration elapsed = clock::now() - start;
            *ms = elapsed.count();
            XPU_LOG("Kernel '%s' took %f ms", type_name<K>(), *ms);
        }

        return 0;
    }

};

} // namespace xpu::detail

#endif
