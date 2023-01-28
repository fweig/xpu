#ifndef TEST_KERNELS_H
#define TEST_KERNELS_H

#include <xpu/device.h>

enum device_funcs {
    ABS,
    ACOS,
    ACOSH,
    ASIN,
    ASINH,
    ATAN2,
    ATAN,
    ATANH,
    CBRT,
    CEIL,
    COPYSIGN,
    COS,
    COSH,
    COSPI,
    ERF,
    ERFC,
    EXP2,
    EXP,
    EXPM1,
    FDIM,
    FLOOR,
    FMA,
    FMOD,
    HYPOT,
    ILOGB,
    ISFINITE,
    ISINF,
    ISNAN,
    J0,
    J1,
    JN,
    LDEXP,
    LLRINT,
    LLROUND,
    LOG,
    LOG10,
    LOG1P,
    LOG2,
    LOGB,
    LRINT,
    LROUND,
    MAX,
    MIN,
    NEARBYINT,
    NORM,
    NORM3D,
    NORM4D,
    POW,
    RCBRT,
    REMAINDER,
    REMQUO_REM,
    REMQUO_QUO,
    RHYPOT,
    RINT,
    RNORM,
    RNORM3D,
    RNORM4D,
    ROUND,
    RSQRT,
    SCALBLN,
    SCALBN,
    SIGNBIT,
    SINCOS_SIN,
    SINCOS_COS,
    SINCOSPI_SIN,
    SINCOSPI_COS,
    SIN,
    SINH,
    SINPI,
    SQRT,
    TAN,
    TANH,
    TGAMMA,
    TRUNC,
    Y0,
    Y1,
    YN,

    ATOMIC_CAS,
    ATOMIC_CAS_BLOCK,
    ATOMIC_ADD,
    ATOMIC_ADD_BLOCK,
    ATOMIC_SUB,
    ATOMIC_SUB_BLOCK,
    ATOMIC_AND,
    ATOMIC_AND_BLOCK,
    ATOMIC_OR,
    ATOMIC_OR_BLOCK,
    ATOMIC_XOR,
    ATOMIC_XOR_BLOCK,

    NUM_DEVICE_FUNCS,
};

union variant {
    bool b;
    int i;
    unsigned int u;
    long long int ll;
    unsigned long long int ull;
    float f;
    double d;
};

struct TestKernels {};

struct float3_ {
    float x;
    float y;
    float z;
};

struct key_value_t {
    unsigned int key;
    unsigned int value;
};

struct test_constants : xpu::constant<TestKernels, float3_> {};

struct get_driver_type : xpu::function<TestKernels> {
    int operator()(xpu::driver_t *);
};

struct empty_kernel : xpu::kernel<TestKernels> {
    using context = xpu::kernel_context<xpu::no_smem>;
    XPU_D void operator()(context &);
};

struct vector_add : xpu::kernel<TestKernels> {
    using context = xpu::kernel_context<xpu::no_smem>;
    XPU_D void operator()(context &, const float *, const float *, float *, int);
};

struct vector_add_timing0 : xpu::kernel<TestKernels> {
    using context = xpu::kernel_context<xpu::no_smem>;
    XPU_D void operator()(context &, const float *, const float *, float *, int);
};

struct vector_add_timing1 : xpu::kernel<TestKernels> {
    using context = xpu::kernel_context<xpu::no_smem>;
    XPU_D void operator()(context &, const float *, const float *, float *, int);
};

struct sort_float : xpu::kernel<TestKernels> {
    using sort_t = xpu::block_sort<float, float, 64, 2>;
    using shared_memory = sort_t::storage_t;
    using context = xpu::kernel_context<shared_memory>;
    XPU_D void operator()(context &, float *, int, float *, float **);
};

struct sort_struct : xpu::kernel<TestKernels> {
    using sort_t = xpu::block_sort<unsigned int, key_value_t, 64, 8>;
    using shared_memory = sort_t::storage_t;
    using context = xpu::kernel_context<shared_memory>;
    XPU_D void operator()(context &, key_value_t *, int, key_value_t *, key_value_t **);
};

struct merge : xpu::kernel<TestKernels> {
    using merge_t = xpu::block_merge<float, block_size::value.x, 8>;
    using shared_memory = merge_t::storage_t;
    using context = xpu::kernel_context<shared_memory>;
    XPU_D void operator()(context &, const float *, size_t, const float *, size_t, float *);
};

struct merge_single : xpu::kernel<TestKernels> {
    using merge_t = xpu::block_merge<float, block_size::value.x, 1>;
    using shared_memory = merge_t::storage_t;
    using context = xpu::kernel_context<shared_memory>;
    XPU_D void operator()(context &, const float *, size_t, const float *, size_t, float *);
};

struct block_scan : xpu::kernel<TestKernels> {
    using scan_t = xpu::block_scan<int, block_size::value.x>;
    using shared_memory = scan_t::storage_t;
    using context = xpu::kernel_context<shared_memory>;
    XPU_D void operator()(context &, int *, int *);
};

struct access_cmem : xpu::kernel<TestKernels> {
    using context = xpu::kernel_context<xpu::no_smem>;
    XPU_D void operator()(context &, float3_ *);
};

struct get_thread_idx_1d : xpu::kernel<TestKernels> {
    using block_size = xpu::block_size<128>;
    using context = xpu::kernel_context<xpu::no_smem>;
    XPU_D void operator()(context &, int *, int *, int *, int *);
};

struct get_thread_idx_2d : xpu::kernel<TestKernels> {
    using block_size = xpu::block_size<32, 8>;
    using context = xpu::kernel_context<xpu::no_smem>;
    XPU_D void operator()(context &, int *, int *, int *, int *);
};

struct get_thread_idx_3d : xpu::kernel<TestKernels> {
    using block_size = xpu::block_size<32, 8, 2>;
    using context = xpu::kernel_context<xpu::no_smem>;
    XPU_D void operator()(context &, int *, int *, int *, int *);
};

struct test_device_funcs : xpu::kernel<TestKernels> {
    using context = xpu::kernel_context<xpu::no_smem>;
    XPU_D void operator()(context &, variant *);
};

#endif
