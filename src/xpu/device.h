#ifndef XPU_DEVICE_H
#define XPU_DEVICE_H

#include "defines.h"
#include "common.h"

#include <type_traits>

#define XPU_IMAGE(image) XPU_DETAIL_IMAGE(image)

#define XPU_EXPORT_CONSTANT(image, type_, name) XPU_DETAIL_EXPORT_CONSTANT(image, type_, name)
#define XPU_EXPORT_FUNC(image, name, ...) XPU_DETAIL_EXPORT_FUNC(image, name, ##__VA_ARGS__)
#define XPU_EXPORT_KERNEL(image, name, ...) XPU_DETAIL_EXPORT_KERNEL(image, name, ##__VA_ARGS__)

#define XPU_CONSTANT(name) XPU_DETAIL_CONSTANT(name)
#define XPU_FUNC(name, ...) XPU_DETAIL_FUNC(name, ##__VA_ARGS__)
#define XPU_FUNC_T(name, ...) XPU_DETAIL_FUNC_T(name, ##__VA_ARGS__)
#define XPU_FUNC_TI(name) XPU_DETAIL_FUNC_TI(name)
#define XPU_FUNC_TS(name, ...) XPU_DETAIL_FUNC_TS(name, ##__VA_ARGS__)
#define XPU_KERNEL(name, shared_memory, ...) XPU_DETAIL_KERNEL(name, shared_memory, ##__VA_ARGS__)
#define XPU_BLOCK_SIZE_1D(kernel, size) XPU_DETAIL_BLOCK_SIZE_1D(kernel, size)
#define XPU_BLOCK_SIZE_2D(kernel, size_x, size_y) XPU_DETAIL_BLOCK_SIZE_2D(kernel, size_x, size_y)
#define XPU_BLOCK_SIZE_3D(kernel, size_x, size_y, size_z) XPU_DETAIL_BLOCK_SIZE_3D(kernel, size_x, size_y, size_z)

#define XPU_ASSERT(x) XPU_DETAIL_ASSERT(x)

#define XPU_T(...) __VA_ARGS__

namespace xpu {

struct thread_idx {
    thread_idx() = delete;
    XPU_D static int x();
    XPU_D static int y();
    XPU_D static int z();
};

struct block_dim {
    block_dim() = delete;
    XPU_D static int x();
    XPU_D static int y();
    XPU_D static int z();};

struct block_idx {
    block_idx() = delete;
    XPU_D static int x();
    XPU_D static int y();
    XPU_D static int z();
};

struct grid_dim {
    grid_dim() = delete;
    XPU_D static int x();
    XPU_D static int y();
    XPU_D static int z();
};

template<typename K>
struct block_size : std::integral_constant<int, 64> {};

struct no_smem {};

template<typename C>
XPU_D const typename C::data_t &cmem() { return C::get(); }

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

template<typename Key, typename KeyValueType, int BlockSize, int ItemsPerThread=8, xpu::driver_t Impl=XPU_COMPILATION_TARGET>
class block_sort {

public:
    struct storage_t {};

    XPU_D block_sort(storage_t &);

    template<typename KeyGetter>
    XPU_D KeyValueType *sort(KeyValueType *vals, size_t N, KeyValueType *buf, KeyGetter &&getKey);
};

template<typename Key, int BlockSize, int ItemsPerThread=8, xpu::driver_t Impl=XPU_COMPILATION_TARGET>
class block_merge {

public:
    struct storage_t {};

    XPU_D block_merge(storage_t &);

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
