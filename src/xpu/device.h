#ifndef XPU_DEVICE_H
#define XPU_DEVICE_H

#include "defines.h"
#include "common.h"

#define _USE_MATH_DEFINES
#include <cmath>
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
#define XPU_BLOCK_SIZE(kernel, size) XPU_DETAIL_BLOCK_SIZE(kernel, size)

#define XPU_ASSERT(x) XPU_DETAIL_ASSERT(x)

#define XPU_T(...) __VA_ARGS__

namespace xpu {

struct thread_idx {
    thread_idx() = delete;
    XPU_D static XPU_FORCE_INLINE int x();
};

struct block_dim {
    block_dim() = delete;
    XPU_D static XPU_FORCE_INLINE int x();
};

struct block_idx {
    block_idx() = delete;
    XPU_D static XPU_FORCE_INLINE int x();
};

struct grid_dim {
    grid_dim() = delete;
    XPU_D static XPU_FORCE_INLINE int x();
};

template<typename K>
struct block_size : std::integral_constant<int, 64> {};

struct no_smem {};

template<typename C>
XPU_D XPU_FORCE_INLINE const typename C::data_t &cmem() { return C::get(); }

XPU_D XPU_FORCE_INLINE constexpr float pi() { return M_PIf32; }
XPU_D XPU_FORCE_INLINE constexpr float deg_to_rad() { return pi() / 180.f; }

XPU_D XPU_FORCE_INLINE   int abs(int x);
XPU_D XPU_FORCE_INLINE float abs(float x);

XPU_D XPU_FORCE_INLINE float acos(float x);

XPU_D XPU_FORCE_INLINE float acosh(float x);

XPU_D XPU_FORCE_INLINE float asin(float x);

XPU_D XPU_FORCE_INLINE float asinh(float x);

XPU_D XPU_FORCE_INLINE float atan2(float y, float x);

XPU_D XPU_FORCE_INLINE float atan(float x);

XPU_D XPU_FORCE_INLINE float atanh(float x);

XPU_D XPU_FORCE_INLINE float cbrt(float x);

XPU_D XPU_FORCE_INLINE float ceil(float x);

XPU_D XPU_FORCE_INLINE float copysign(float x, float y);

XPU_D XPU_FORCE_INLINE float cos(float x);

XPU_D XPU_FORCE_INLINE float cosh(float x);

XPU_D XPU_FORCE_INLINE float cospi(float x);

// Not supported by HIP or c++11
// XPU_D XPU_FORCE_INLINE float cyl_bessel_i0f(float x);
// XPU_D XPU_FORCE_INLINE float cyl_bessel_i1f(float x);

XPU_D XPU_FORCE_INLINE float erf(float x);

// Not supported by c++ stdlib
// XPU_D XPU_FORCE_INLINE float erfinv(float y);

XPU_D XPU_FORCE_INLINE float erfc(float x);

// Not supported by c++ stdlib
// XPU_D XPU_FORCE_INLINE float erfcinv(float y);

// Not supported by c++ stdlib
// XPU_D XPU_FORCE_INLINE float erfcx(float x);

XPU_D XPU_FORCE_INLINE float exp2(float x);

XPU_D XPU_FORCE_INLINE float exp(float x);

XPU_D XPU_FORCE_INLINE float expm1(float x);

XPU_D XPU_FORCE_INLINE float fdim(float x, float y);

XPU_D XPU_FORCE_INLINE float floor(float x);

XPU_D XPU_FORCE_INLINE float fma(float x, float y, float z);

XPU_D XPU_FORCE_INLINE float fmod(float x, float y);

// Not supported by HIP (as of 4.5)
// XPU_D XPU_FORCE_INLINE float frexp(float x, int *nptr);

XPU_D XPU_FORCE_INLINE float hypot(float x, float y);

XPU_D XPU_FORCE_INLINE float ilogb(float x);

XPU_D XPU_FORCE_INLINE bool isfinite(float a);

XPU_D XPU_FORCE_INLINE bool isinf(float a);

XPU_D XPU_FORCE_INLINE bool isnan(float a);

XPU_D XPU_FORCE_INLINE float j0(float x);

XPU_D XPU_FORCE_INLINE float j1(float x);

XPU_D XPU_FORCE_INLINE float jn(int n, float x);

XPU_D XPU_FORCE_INLINE float ldexp(float x, int exp);

// single-precision version not supported by HIP (as of 4.5)
// XPU_D XPU_FORCE_INLINE float lgamma(float x);

XPU_D XPU_FORCE_INLINE long long int llrint(float x);

XPU_D XPU_FORCE_INLINE long long int llround(float x);

XPU_D XPU_FORCE_INLINE float log(float x);

XPU_D XPU_FORCE_INLINE float log10(float x);

XPU_D XPU_FORCE_INLINE float log1p(float x);

XPU_D XPU_FORCE_INLINE float log2(float x);

XPU_D XPU_FORCE_INLINE float logb(float x);

XPU_D XPU_FORCE_INLINE long int lrint(float x);

XPU_D XPU_FORCE_INLINE long int lround(float x);

XPU_D XPU_FORCE_INLINE                    int max(int a, int b);
XPU_D XPU_FORCE_INLINE           unsigned int max(unsigned int a, unsigned int b);
XPU_D XPU_FORCE_INLINE          long long int max(long long int a, long long int b);
XPU_D XPU_FORCE_INLINE unsigned long long int max(unsigned long long int a, unsigned long long int b);
XPU_D XPU_FORCE_INLINE                  float max(float a, float b);

XPU_D XPU_FORCE_INLINE                    int min(int a, int b);
XPU_D XPU_FORCE_INLINE           unsigned int min(unsigned int a, unsigned int b);
XPU_D XPU_FORCE_INLINE          long long int min(long long int a, long long int b);
XPU_D XPU_FORCE_INLINE unsigned long long int min(unsigned long long int a, unsigned long long int b);
XPU_D XPU_FORCE_INLINE                  float min(float a, float b);

// Not supported by HIP (as of 4.5)
// XPU_D XPU_FORCE_INLINE float modf(float x, float *iptr);

XPU_D XPU_FORCE_INLINE float nan(const char *tagp);

XPU_D XPU_FORCE_INLINE float nearbyint(float x);

// Not supported by HIP (as of 4.5)
// XPU_D XPU_FORCE_INLINE float nextafter(float x, float y);

XPU_D XPU_FORCE_INLINE float norm(int dim, const float *a);
XPU_D XPU_FORCE_INLINE float norm3d(float a, float b, float c);
XPU_D XPU_FORCE_INLINE float norm4d(float a, float b, float c, float d);

// Not supported by c++ stdlib (TODO: provide own implementation)
// XPU_D XPU_FORCE_INLINE float normcdf(float y);
// XPU_D XPU_FORCE_INLINE float normcdfinv(float y);

XPU_D XPU_FORCE_INLINE float pow(float x, float y);

XPU_D XPU_FORCE_INLINE float rcbrt(float x);

XPU_D XPU_FORCE_INLINE float remainder(float x, float y);

XPU_D XPU_FORCE_INLINE float remquo(float x, float y, int *quo);

XPU_D XPU_FORCE_INLINE float rhypot(float x, float y);

XPU_D XPU_FORCE_INLINE float rint(float x);

XPU_D XPU_FORCE_INLINE float rnorm(int dim, const float *a);
XPU_D XPU_FORCE_INLINE float rnorm3d(float a, float b, float c);
XPU_D XPU_FORCE_INLINE float rnorm4d(float a, float b, float c, float d);

XPU_D XPU_FORCE_INLINE float round(float x);

XPU_D XPU_FORCE_INLINE float rsqrt(float x);

XPU_D XPU_FORCE_INLINE float scalbln(float x, long int n);
XPU_D XPU_FORCE_INLINE float scalbn(float x, int n);

XPU_D XPU_FORCE_INLINE bool signbit(float a);

XPU_D XPU_FORCE_INLINE void sincos(float x, float *sptr, float *cptr);

XPU_D XPU_FORCE_INLINE void sincospi(float x, float *sptr, float *cptr);

XPU_D XPU_FORCE_INLINE float sin(float x);

XPU_D XPU_FORCE_INLINE float sinh(float x);

XPU_D XPU_FORCE_INLINE float sinpi(float x);

XPU_D XPU_FORCE_INLINE float sqrt(float x);

XPU_D XPU_FORCE_INLINE float tan(float x);

XPU_D XPU_FORCE_INLINE float tanh(float x);

XPU_D XPU_FORCE_INLINE float tgamma(float x);

XPU_D XPU_FORCE_INLINE float trunc(float x);

XPU_D XPU_FORCE_INLINE float y0(float x);
XPU_D XPU_FORCE_INLINE float y1(float x);
XPU_D XPU_FORCE_INLINE float yn(int n, float x);

XPU_D XPU_FORCE_INLINE          int atomic_cas(int *addr, int compare, int val);
XPU_D XPU_FORCE_INLINE unsigned int atomic_cas(unsigned int *addr, unsigned int compare, unsigned int val);
XPU_D XPU_FORCE_INLINE        float atomic_cas(float *addr, float compare, float val);
XPU_D XPU_FORCE_INLINE          int atomic_cas_block(int *addr, int compare, int val);
XPU_D XPU_FORCE_INLINE unsigned int atomic_cas_block(unsigned int *addr, unsigned int compare, unsigned int val);
XPU_D XPU_FORCE_INLINE        float atomic_cas_block(float *addr, float compare, float val);

XPU_D XPU_FORCE_INLINE          int atomic_add(int *addr, int val);
XPU_D XPU_FORCE_INLINE unsigned int atomic_add(unsigned int *addr, unsigned int val);
XPU_D XPU_FORCE_INLINE        float atomic_add(float *addr, float val);
XPU_D XPU_FORCE_INLINE          int atomic_add_block(int *addr, int val);
XPU_D XPU_FORCE_INLINE unsigned int atomic_add_block(unsigned int *addr, unsigned int val);
XPU_D XPU_FORCE_INLINE        float atomic_add_block(float *addr, float val);

XPU_D XPU_FORCE_INLINE          int atomic_sub(int *addr, int val);
XPU_D XPU_FORCE_INLINE unsigned int atomic_sub(unsigned int *addr, unsigned int val);
XPU_D XPU_FORCE_INLINE          int atomic_sub_block(int *addr, int val);
XPU_D XPU_FORCE_INLINE unsigned int atomic_sub_block(unsigned int *addr, unsigned int val);

XPU_D XPU_FORCE_INLINE          int atomic_and(int *addr, int val);
XPU_D XPU_FORCE_INLINE unsigned int atomic_and(unsigned int *addr, unsigned int val);
XPU_D XPU_FORCE_INLINE          int atomic_and_block(int *addr, int val);
XPU_D XPU_FORCE_INLINE unsigned int atomic_and_block(unsigned int *addr, unsigned int val);

XPU_D XPU_FORCE_INLINE          int atomic_or(int *addr, int val);
XPU_D XPU_FORCE_INLINE unsigned int atomic_or(unsigned int *addr, unsigned int val);
XPU_D XPU_FORCE_INLINE          int atomic_or_block(int *addr, int val);
XPU_D XPU_FORCE_INLINE unsigned int atomic_or_block(unsigned int *addr, unsigned int val);

XPU_D XPU_FORCE_INLINE          int atomic_xor(int *addr, int val);
XPU_D XPU_FORCE_INLINE unsigned int atomic_xor(unsigned int *addr, unsigned int val);
XPU_D XPU_FORCE_INLINE          int atomic_xor_block(int *addr, int val);
XPU_D XPU_FORCE_INLINE unsigned int atomic_xor_block(unsigned int *addr, unsigned int val);

XPU_D XPU_FORCE_INLINE void barrier();

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

#if XPU_IS_HIP_CUDA
#include "driver/hip_cuda/device.h"
#else // CPU
#include "driver/cpu/device.h"
#endif

#endif
