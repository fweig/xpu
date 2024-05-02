/**
 * @file device.h
 * @brief Device-side API.
 *
 * This file contains the device-side API of xpu.
 *
 * Include as `#include <xpu/device.h>`.
 *
 * @note This file is safe to include from both host and device code.
 */
#ifndef XPU_DEVICE_H
#define XPU_DEVICE_H

#include "defines.h"
#include "common.h"

#include "detail/common.h"
#include "detail/type_info.h"

#if XPU_IS_CPU
#include "detail/platform/cpu/cmem_impl.h"
#include "detail/platform/cpu/tpos_impl.h"
#elif XPU_IS_HIP_CUDA
#include "detail/platform/hip_cuda/cmem_impl.h"
#include "detail/platform/hip_cuda/tpos_impl.h"
#elif XPU_IS_SYCL
#include "detail/platform/sycl/cmem_impl.h"
#include "detail/platform/sycl/tpos_impl.h"
#else
#error "Unsupported XPU target"
#endif

#include <type_traits>

#define XPU_IMAGE(image) XPU_DETAIL_IMAGE(image)
#define XPU_EXPORT(obj) XPU_DETAIL_EXPORT(obj)

#define XPU_ASSERT(x) XPU_DETAIL_ASSERT(x)

namespace xpu {

constexpr inline driver_t compilation_target = XPU_DETAIL_COMPILATION_TARGET;

struct device_image : detail::device_image {};

template<int X, int Y = -1, int Z = -1>
struct block_size {
    static inline constexpr xpu::dim value{X, Y, Z};
};

struct no_smem {};

/**
 * @brief OpenMP schedule types. Used for specifying the schedule type for kernels.
 */
enum schedule_t {
    schedule_static, // Avoid conflicts with C++ 'static' keyword
    schedule_dynamic,
};

/**
 * @brief OpenMP settings for kernels.
 */
template<schedule_t Schedule = schedule_static, size_t ChunkSize = 0>
struct openmp_settings {

    /**
     * @brief OpenMP schedule type.
     */
    static constexpr schedule_t schedule = Schedule;

    /**
     * @brief Chunk size. Use 0 for default value from OpenMP.
     */
    static constexpr size_t chunk_size = ChunkSize;
};

class tpos {

public:
    XPU_D int thread_idx_x() const { return m_impl.thread_idx_x(); }
    XPU_D int thread_idx_y() const { return m_impl.thread_idx_y(); }
    XPU_D int thread_idx_z() const { return m_impl.thread_idx_z(); }

    XPU_D int block_dim_x() const { return m_impl.block_dim_x(); }
    XPU_D int block_dim_y() const { return m_impl.block_dim_y(); }
    XPU_D int block_dim_z() const { return m_impl.block_dim_z(); }

    XPU_D int block_idx_x() const { return m_impl.block_idx_x(); }
    XPU_D int block_idx_y() const { return m_impl.block_idx_y(); }
    XPU_D int block_idx_z() const { return m_impl.block_idx_z(); }

    XPU_D int grid_dim_x() const { return m_impl.grid_dim_x(); }
    XPU_D int grid_dim_y() const { return m_impl.grid_dim_y(); }
    XPU_D int grid_dim_z() const { return m_impl.grid_dim_z(); }

private:
    detail::tpos_impl m_impl;

public:
    template<typename... Args>
    XPU_D tpos(detail::internal_ctor_t, Args &&... args)
        : m_impl(std::forward<Args>(args)...) {}

    XPU_D detail::tpos_impl &impl(detail::internal_fn_t) { return m_impl; }
};

template<typename... Constants>
class cmem {

public:
    template<typename Constant>
    XPU_D const typename Constant::data_t &get() const { return m_impl.template get<Constant>(); }

private:
    detail::cmem_impl<Constants...> m_impl;

public:
    template<typename... Args>
    XPU_D cmem(detail::internal_ctor_t, Args &&... args)
        : m_impl(std::forward<Args>(args)...) {}

};


template<typename Image>
struct kernel : detail::action<Image, detail::kernel_tag> {
    // Defaults
    using block_size = xpu::block_size<64>;
    using constants = cmem<>;
    using shared_memory = no_smem;

    /**
     * @brief OpenMP settings for the kernel.
     * @details By default kernels will use static scheduling with the default chunk size.
     */
    using openmp = openmp_settings<schedule_static, 0>;
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

    XPU_D       shared_memory &smem()       { return m_smem; }
    XPU_D const shared_memory &smem() const { return m_smem; }

    /**
     * Shortcut to access a constant from constant memory.
     */
    template<typename C>
    XPU_D const typename C::data_t &cmem() const { return m_cmem.template get<C>(); }

    /**
     * Access the constant memory.
     */
    XPU_D const constants &cmem() const { return m_cmem; }

    /**
     * Shortcut to access thread position in x dimension.
     * Identical to pos().thread_idx_x().
     */
    XPU_D int thread_idx_x() const { return m_pos.thread_idx_x(); }

    /**
     * Shortcut to access thread position in y dimension.
     * Identical to pos().thread_idx_y().
     */
    XPU_D int thread_idx_y() const { return m_pos.thread_idx_y(); }

    /**
     * Shortcut to access thread position in z dimension.
     * Identical to pos().thread_idx_z().
     */
    XPU_D int thread_idx_z() const { return m_pos.thread_idx_z(); }

    /**
     * Shortcut to access block size in x dimension.
     * Identical to pos().block_dim_x().
     */
    XPU_D int block_dim_x() const { return m_pos.block_dim_x(); }

    /**
     * Shortcut to access block size in y dimension.
     * Identical to pos().block_dim_y().
     */
    XPU_D int block_dim_y() const { return m_pos.block_dim_y(); }

    /**
     * Shortcut to access block size in z dimension.
     * Identical to pos().block_dim_z().
     */
    XPU_D int block_dim_z() const { return m_pos.block_dim_z(); }

    /**
     * Shortcut to access block position in x dimension.
     * Identical to pos().block_idx_x().
     */
    XPU_D int block_idx_x() const { return m_pos.block_idx_x(); }

    /**
     * Shortcut to access block position in y dimension.
     * Identical to pos().block_idx_y().
     */
    XPU_D int block_idx_y() const { return m_pos.block_idx_y(); }

    /**
     * Shortcut to access block position in z dimension.
     * Identical to pos().block_idx_z().
     */
    XPU_D int block_idx_z() const { return m_pos.block_idx_z(); }

    /**
     * Shortcut to access grid size in x dimension.
     * Identical to pos().grid_dim_x().
     */
    XPU_D int grid_dim_x() const { return m_pos.grid_dim_x(); }

    /**
     * Shortcut to access grid size in y dimension.
     * Identical to pos().grid_dim_y().
     */
    XPU_D int grid_dim_y() const { return m_pos.grid_dim_y(); }

    /**
     * Shortcut to access grid size in z dimension.
     * Identical to pos().grid_dim_z().
     */
    XPU_D int grid_dim_z() const { return m_pos.grid_dim_z(); }

    XPU_D       tpos &pos()       { return m_pos; }
    XPU_D const tpos &pos() const { return m_pos; }

private:
    tpos          &m_pos;
    shared_memory &m_smem;
    const constants &m_cmem;

public:
    XPU_D kernel_context(detail::internal_ctor_t, tpos &pos, shared_memory &smem, const constants &cmem)
        : m_pos(pos)
        , m_smem(smem)
        , m_cmem(cmem) {}

};

/**
 * Create a view of a buffer.
 * The view is a pointer to the buffer's data and a size.
 * This class should be used to access a buffer in device code only.
 * To access buffer-data on the host, use 'h_view' instead.
 */
template<typename T>
class view {

public:
    /**
     * Create an empty view.
     */
    view() = default;

    /**
     * Create a view from a pointer and a size.
    */
    XPU_D view(T *data, size_t size);

    /**
     * Create a view from a buffer and a size.
     */
    XPU_D view(buffer<T> &buffer, size_t size);

    /**
     * Return a pointer to the view's data.
     */
    XPU_D T *data() const { return m_data; }

    /**
     * Return the view's size.
     */
    XPU_D size_t size() const { return m_size; }

    /**
     * Check if the view is empty.
     */
    XPU_D bool empty() const { return m_size == 0; }

    XPU_D       T &operator[](size_t idx);
    XPU_D const T &operator[](size_t idx) const;

    XPU_D       T &at(size_t idx);
    XPU_D const T &at(size_t idx) const;

private:
    T *m_data = nullptr;
    size_t m_size = 0;

};

// =================================================================================================
// Math functions
//
// Interface is based on CUDA math functions
// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE
//
// Note: Functions not supported in SYCL or HIP were removed.
// =================================================================================================

XPU_D constexpr float pi();
XPU_D constexpr float pi_2();
XPU_D constexpr float pi_4();
XPU_D constexpr float deg_to_rad();
XPU_D constexpr float sqrt2();

XPU_D   int abs(int x);
XPU_D float abs(float x);

XPU_D float acos(float x);
XPU_D float acosh(float x);
XPU_D float acospi(float x);

XPU_D float asin(float x);
XPU_D float asinh(float x);
XPU_D float asinpi(float x);

XPU_D float atan(float x);
XPU_D float atan2(float y, float x);
XPU_D float atanh(float x);
XPU_D float atanpi(float x);
XPU_D float atan2pi(float y, float x);

XPU_D float cbrt(float x);

XPU_D float ceil(float x);

XPU_D float copysign(float x, float y);

XPU_D float cos(float x);
XPU_D float cosh(float x);
XPU_D float cospi(float x);

// Not supported by HIP or c++11 or sycl
// XPU_D float cyl_bessel_i0f(float x);
// XPU_D float cyl_bessel_i1f(float x);

XPU_D float erf(float x);

// Not supported by c++ stdlib
// XPU_D float erfinv(float y);

XPU_D float erfc(float x);

// Not supported by c++ stdlib or sycl
// XPU_D float erfcinv(float y);

// Not supported by c++ stdlib or sycl
// XPU_D float erfcx(float x);

XPU_D float exp(float x);
XPU_D float exp2(float x);
XPU_D float exp10(float x);
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

// Not supported by SYCL
// XPU_D float j0(float x);
// XPU_D float j1(float x);
// XPU_D float jn(int n, float x);

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

// No supported by sycl
// XPU_D float nearbyint(float x);

// Not supported by HIP (as of 4.5)
// XPU_D float nextafter(float x, float y);

// Not supported by SYCL
// XPU_D float norm(int dim, const float *a);
XPU_D float norm3d(float a, float b, float c);
XPU_D float norm4d(float a, float b, float c, float d);

// Not supported by c++ stdlib (TODO: provide own implementation?)
// XPU_D float normcdf(float y);
// XPU_D float normcdfinv(float y);

XPU_D float pow(float x, float y);

// Not supported by CUDA
// XPU_D float pown(float x, int n);
// XPU_D float powr(float x, float y);

XPU_D float rcbrt(float x);

XPU_D float remainder(float x, float y);

XPU_D float remquo(float x, float y, int *quo);

XPU_D float rint(float x);

XPU_D float rhypot(float x, float y);

// Not supported by SYCL
// XPU_D float rnorm(int dim, const float *a);
XPU_D float rnorm3d(float a, float b, float c);
XPU_D float rnorm4d(float a, float b, float c, float d);

XPU_D float round(float x);

XPU_D float rsqrt(float x);

// Not supported by SYCL
// XPU_D float scalbln(float x, long int n);
// XPU_D float scalbn(float x, int n);

XPU_D bool signbit(float a);

XPU_D void sincos(float x, float *sptr, float *cptr);
XPU_D void sincospi(float x, float *sptr, float *cptr);

XPU_D float sin(float x);
XPU_D float sinh(float x);
XPU_D float sinpi(float x);

XPU_D float sqrt(float x);

XPU_D float tan(float x);
XPU_D float tanh(float x);
XPU_D float tanpi(float x);

XPU_D float tanpi(float x);

XPU_D float tgamma(float x);

XPU_D float trunc(float x);

// Not supported by SYCL
// XPU_D float y0(float x);
// XPU_D float y1(float x);
// XPU_D float yn(int n, float x);

XPU_D          int atomic_cas(int *addr, int compare, int val);
XPU_D unsigned int atomic_cas(unsigned int *addr, unsigned int compare, unsigned int val);
XPU_D        float atomic_cas(float *addr, float compare, float val);
XPU_D          int atomic_cas_block(int *addr, int compare, int val);
XPU_D unsigned int atomic_cas_block(unsigned int *addr, unsigned int compare, unsigned int val);
XPU_D        float atomic_cas_block(float *addr, float compare, float val);

XPU_D          int atomic_add(int *addr, int val);
XPU_D unsigned int atomic_add(unsigned int *addr, unsigned int val);
XPU_D        float atomic_add(float *addr, float val);
XPU_D          int atomic_add_block(int *addr, int val);
XPU_D unsigned int atomic_add_block(unsigned int *addr, unsigned int val);
XPU_D        float atomic_add_block(float *addr, float val);

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

XPU_D int float_as_int(float val);
XPU_D float int_as_float(int val);


/**
 * @brief Sync all threads in a block.
 */
XPU_D void barrier(tpos &);

/**
 * @brief Sync all threads in a block.
 *
 * @note This function is a shortcut for `barrier(ctx.pos())`.
 */
template <typename ContextT>
XPU_D void barrier(ContextT &ctx) { barrier(ctx.pos()); }

/**
 * @brief Parallel scan inside a block.
 */
template<typename T, int BlockSize, xpu::driver_t Impl=XPU_COMPILATION_TARGET>
class block_scan {

public:
    /**
     * @brief Temporary storage for the block scan. Should be allocated in shared memory.
     */
    struct storage_t {};

    /**
     * @brief Construct a block scan object.
     *
     * @param ctx Kernel context.
     * @param storage Temporary storage for the block scan.
     *
     * @note This is a shortcut for `block_scan(ctx.pos(), storage)`.
     * @see block_scan::storage_t
     */
    template<typename ContextT>
    XPU_D block_scan(ContextT &ctx, storage_t &storage);

    /**
     * @brief Construct a block scan object.
     *
     * @param pos Thread position.
     * @param storage Temporary storage for the block scan.
     *
     * @see block_scan::storage_t
     */
    XPU_D block_scan(tpos &pos, storage_t &storage);

    XPU_D void exclusive_sum(T input, T &output);

    template<typename ScanOp>
    XPU_D void exclusive_sum(T input, T &output, T initial_value, ScanOp scan_op);

    XPU_D void inclusive_sum(T input, T &output);

    template<typename ScanOp>
    XPU_D void inclusive_sum(T input, T &output, T initial_value, ScanOp scan_op);
};

template <typename T, int BlockSize, xpu::driver_t Impl = XPU_COMPILATION_TARGET>
class block_reduce
{

public:
    struct storage_t {};

    template<typename ContextT>
    XPU_D block_reduce(ContextT &ctx, storage_t &storage);

    XPU_D block_reduce(tpos &, storage_t &);

    XPU_D T sum(T input);

    template<typename ReduceOp>
    XPU_D T reduce(T input, ReduceOp reduce_op);
};

template <typename Key, typename KeyValueType, int BlockSize, int ItemsPerThread = 8, xpu::driver_t Impl = XPU_COMPILATION_TARGET>
class block_sort
{

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

#if XPU_IS_HIP_CUDA
#include "detail/platform/hip_cuda/device.h"
#elif XPU_IS_SYCL
#include "detail/platform/sycl/device.h"
#elif XPU_IS_CPU
#include "detail/platform/cpu/device.h"
#else
#error "Unknown XPU driver."
#endif

#include "detail/constants.h"
#include "detail/view_impl.h"

#endif
