#ifndef XPU_DEVICE_H
#define XPU_DEVICE_H

#include "defines.h"
#include "common.h"
#include "detail/common.h"
#include "detail/type_info.h"

#include <type_traits>

#define XPU_IMAGE(image) XPU_DETAIL_IMAGE(image)
#define XPU_EXPORT(obj) XPU_DETAIL_EXPORT(obj)
#define XPU_CONSTANT(name) XPU_DETAIL_CONSTANT(name)

#define XPU_ASSERT(x) XPU_DETAIL_ASSERT(x)

#define XPU_T(...) __VA_ARGS__

namespace xpu {

template<int X, int Y = -1, int Z = -1>
struct block_size {
    static inline constexpr xpu::dim value{X, Y, Z};
};

struct no_smem {};

template<xpu::driver_t Impl=XPU_COMPILATION_TARGET>
class tpos_impl {

public:
    XPU_D int thread_idx_x() const;
    XPU_D int thread_idx_y() const;
    XPU_D int thread_idx_z() const;

    XPU_D int block_dim_x() const;
    XPU_D int block_dim_y() const;
    XPU_D int block_dim_z() const;

    XPU_D int block_idx_x() const;
    XPU_D int block_idx_y() const;
    XPU_D int block_idx_z() const;

    XPU_D int grid_dim_x() const;
    XPU_D int grid_dim_y() const;
    XPU_D int grid_dim_z() const;
};

using tpos = tpos_impl<>;

template<xpu::driver_t, typename... Constants>
class cmem_impl {

public:
    template<typename Constant>
    XPU_D const typename Constant::data_t &get() const;

};

template<typename... Constants>
using cmem = cmem_impl<XPU_COMPILATION_TARGET, Constants...>;

template<typename Image>
struct kernel : detail::action<Image, detail::kernel_tag> {
    // Defaults
    using block_size = xpu::block_size<64>;
    using constants = cmem<>;
    using shared_memory = no_smem;
};

template<typename Image>
struct function : detail::action<Image, detail::function_tag> {
};

template<typename Image, typename Data>
struct constant : detail::action<Image, detail::constant_tag> {
    using data_t = Data;
};

template<typename SharedMemory = xpu::no_smem, typename Constants = xpu::cmem<>>
class kernel_context {

public:
    using shared_memory = SharedMemory;
    using constants = Constants;

    XPU_D kernel_context(tpos &pos, shared_memory &smem, constants &cmem)
        : m_pos(pos)
        , m_smem(smem)
        , m_cmem(cmem) {}

    XPU_D       shared_memory &smem()       { return m_smem; }
    XPU_D const shared_memory &smem() const { return m_smem; }

    XPU_D       constants &cmem()       { return m_cmem; }
    XPU_D const constants &cmem() const { return m_cmem; }

    XPU_D       tpos &pos()       { return m_pos; }
    XPU_D const tpos &pos() const { return m_pos; }

private:
    tpos          &m_pos;
    shared_memory &m_smem;
    constants     &m_cmem;

};

XPU_D constexpr float pi();
XPU_D constexpr float pi_2();
XPU_D constexpr float pi_4();
XPU_D constexpr float deg_to_rad();
XPU_D constexpr float sqrt2();

XPU_D   int abs(int x);
XPU_D float abs(float x);

XPU_D float acos(float x);

XPU_D float acosh(float x);

XPU_D float asin(float x);

XPU_D float asinh(float x);

XPU_D float atan2(float y, float x);

XPU_D float atan(float x);

XPU_D float atanh(float x);

XPU_D float cbrt(float x);

XPU_D float ceil(float x);

XPU_D float copysign(float x, float y);

XPU_D float cos(float x);

XPU_D float cosh(float x);

XPU_D float cospi(float x);

// Not supported by HIP or c++11
// XPU_D float cyl_bessel_i0f(float x);
// XPU_D float cyl_bessel_i1f(float x);

XPU_D float erf(float x);

// Not supported by c++ stdlib
// XPU_D float erfinv(float y);

XPU_D float erfc(float x);

// Not supported by c++ stdlib
// XPU_D float erfcinv(float y);

// Not supported by c++ stdlib
// XPU_D float erfcx(float x);

XPU_D float exp2(float x);

XPU_D float exp(float x);

XPU_D float expm1(float x);

XPU_D float fdim(float x, float y);

XPU_D float floor(float x);

XPU_D float fma(float x, float y, float z);

XPU_D float fmod(float x, float y);

// Not supported by HIP (as of 4.5)
// XPU_D float frexp(float x, int *nptr);

XPU_D float hypot(float x, float y);

XPU_D int ilogb(float x);

XPU_D bool isfinite(float a);

XPU_D bool isinf(float a);

XPU_D bool isnan(float a);

XPU_D float j0(float x);

XPU_D float j1(float x);

XPU_D float jn(int n, float x);

XPU_D float ldexp(float x, int exp);

// single-precision version not supported by HIP (as of 4.5)
// XPU_D float lgamma(float x);

XPU_D long long int llrint(float x);

XPU_D long long int llround(float x);

XPU_D float log(float x);

XPU_D float log10(float x);

XPU_D float log1p(float x);

XPU_D float log2(float x);

XPU_D float logb(float x);

XPU_D long int lrint(float x);

XPU_D long int lround(float x);

XPU_D                    int max(int a, int b);
XPU_D           unsigned int max(unsigned int a, unsigned int b);
XPU_D          long long int max(long long int a, long long int b);
XPU_D unsigned long long int max(unsigned long long int a, unsigned long long int b);
XPU_D                  float max(float a, float b);

XPU_D                    int min(int a, int b);
XPU_D           unsigned int min(unsigned int a, unsigned int b);
XPU_D          long long int min(long long int a, long long int b);
XPU_D unsigned long long int min(unsigned long long int a, unsigned long long int b);
XPU_D                  float min(float a, float b);

// Not supported by HIP (as of 4.5)
// XPU_D float modf(float x, float *iptr);

XPU_D float nan(const char *tagp);

XPU_D float nearbyint(float x);

// Not supported by HIP (as of 4.5)
// XPU_D float nextafter(float x, float y);

XPU_D float norm(int dim, const float *a);
XPU_D float norm3d(float a, float b, float c);
XPU_D float norm4d(float a, float b, float c, float d);

// Not supported by c++ stdlib (TODO: provide own implementation)
// XPU_D float normcdf(float y);
// XPU_D float normcdfinv(float y);

XPU_D float pow(float x, float y);

XPU_D float rcbrt(float x);

XPU_D float remainder(float x, float y);

XPU_D float remquo(float x, float y, int *quo);

XPU_D float rhypot(float x, float y);

XPU_D float rint(float x);

XPU_D float rnorm(int dim, const float *a);
XPU_D float rnorm3d(float a, float b, float c);
XPU_D float rnorm4d(float a, float b, float c, float d);

XPU_D float round(float x);

XPU_D float rsqrt(float x);

XPU_D float scalbln(float x, long int n);
XPU_D float scalbn(float x, int n);

XPU_D bool signbit(float a);

XPU_D void sincos(float x, float *sptr, float *cptr);

XPU_D void sincospi(float x, float *sptr, float *cptr);

XPU_D float sin(float x);

XPU_D float sinh(float x);

XPU_D float sinpi(float x);

XPU_D float sqrt(float x);

XPU_D float tan(float x);

XPU_D float tanh(float x);

XPU_D float tgamma(float x);

XPU_D float trunc(float x);

XPU_D float y0(float x);
XPU_D float y1(float x);
XPU_D float yn(int n, float x);

XPU_D          int atomic_cas(int *addr, int compare, int val);
XPU_D unsigned int atomic_cas(unsigned int *addr, unsigned int compare, unsigned int val);
XPU_D          int atomic_cas_block(int *addr, int compare, int val);
XPU_D unsigned int atomic_cas_block(unsigned int *addr, unsigned int compare, unsigned int val);

XPU_D          int atomic_add(int *addr, int val);
XPU_D unsigned int atomic_add(unsigned int *addr, unsigned int val);
XPU_D          int atomic_add_block(int *addr, int val);
XPU_D unsigned int atomic_add_block(unsigned int *addr, unsigned int val);

XPU_D          int atomic_sub(int *addr, int val);
XPU_D unsigned int atomic_sub(unsigned int *addr, unsigned int val);
XPU_D          int atomic_sub_block(int *addr, int val);
XPU_D unsigned int atomic_sub_block(unsigned int *addr, unsigned int val);

XPU_D          int atomic_and(int *addr, int val);
XPU_D unsigned int atomic_and(unsigned int *addr, unsigned int val);
XPU_D          int atomic_and_block(int *addr, int val);
XPU_D unsigned int atomic_and_block(unsigned int *addr, unsigned int val);

XPU_D          int atomic_or(int *addr, int val);
XPU_D unsigned int atomic_or(unsigned int *addr, unsigned int val);
XPU_D          int atomic_or_block(int *addr, int val);
XPU_D unsigned int atomic_or_block(unsigned int *addr, unsigned int val);

XPU_D          int atomic_xor(int *addr, int val);
XPU_D unsigned int atomic_xor(unsigned int *addr, unsigned int val);
XPU_D          int atomic_xor_block(int *addr, int val);
XPU_D unsigned int atomic_xor_block(unsigned int *addr, unsigned int val);

XPU_D void barrier();

XPU_D int float_as_int(float val);
XPU_D float int_as_float(int val);


template<typename T, int BlockSize, xpu::driver_t Impl=XPU_COMPILATION_TARGET>
class block_scan {

public:
    struct storage_t {};

    XPU_D block_scan(storage_t &);

    XPU_D void exclusive_sum(T input, T &output);

    template<typename ScanOp>
    XPU_D void exclusive_sum(T input, T &output, T initial_value, ScanOp scan_op);

    template<int ItemsPerThread>
    XPU_D void exclusive_sum(T(&input)[ItemsPerThread], T(&output)[ItemsPerThread]);

    template<int ItemsPerThread, typename ScanOp>
    XPU_D void exclusive_sum(T(&input)[ItemsPerThread], T(&output)[ItemsPerThread], ScanOp scan_op);

    XPU_D void inclusive_sum(T input, T &output);

    template<typename ScanOp>
    XPU_D void inclusive_sum(T input, T &output, T initial_value, ScanOp scan_op);

    template<int ItemsPerThread>
    XPU_D void inclusive_sum(T(&input)[ItemsPerThread], T(&output)[ItemsPerThread]);

    template<int ItemsPerThread, typename ScanOp>
    XPU_D void inclusive_sum(T(&input)[ItemsPerThread], T(&output)[ItemsPerThread], T initial_value, ScanOp scan_op);
};

template<typename Key, typename KeyValueType, int BlockSize, int ItemsPerThread=8, xpu::driver_t Impl=XPU_COMPILATION_TARGET>
class block_sort {

public:
    struct storage_t {};

    XPU_D block_sort(tpos &, storage_t &);

    template<typename KeyGetter>
    XPU_D KeyValueType *sort(KeyValueType *vals, size_t N, KeyValueType *buf, KeyGetter &&getKey);
};

template<typename Key, int BlockSize, int ItemsPerThread=8, xpu::driver_t Impl=XPU_COMPILATION_TARGET>
class block_merge {

public:
    struct storage_t {};

    XPU_D block_merge(tpos &, storage_t &);

    template<typename Compare>
    XPU_D void merge(const Key *a, size_t size_a, const Key *b, size_t size_b, Key *dst, Compare &&);

};

} // namespace xpu

#include "detail/dynamic_loader.h"
#include "detail/constants.h"

#if XPU_IS_HIP_CUDA
#include "driver/hip_cuda/device.h"
#else // CPU
#include "driver/cpu/device.h"
#endif

#endif
