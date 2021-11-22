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

XPU_EXPORT_CONSTANT(TestKernels, float3_, test_constants);

XPU_EXPORT_KERNEL(TestKernels, empty_kernel);
XPU_EXPORT_KERNEL(TestKernels, vector_add, const float *, const float *, float *, int);
XPU_EXPORT_KERNEL(TestKernels, vector_add_timing, const float *, const float *, float *, int);
XPU_EXPORT_KERNEL(TestKernels, sort_float, float *, int, float *, float **);
XPU_EXPORT_KERNEL(TestKernels, sort_struct, key_value_t *, int, key_value_t *, key_value_t **);
XPU_EXPORT_KERNEL(TestKernels, merge, const float *, size_t, const float *, size_t, float *);
XPU_EXPORT_KERNEL(TestKernels, merge_single, const float *, size_t, const float *, size_t, float *);
XPU_EXPORT_KERNEL(TestKernels, access_cmem, float3_ *);
XPU_EXPORT_KERNEL(TestKernels, get_thread_idx, int *);

XPU_EXPORT_KERNEL(TestKernels, test_device_funcs, variant *);

#endif
