#include "TestKernels.h"

XPU_IMAGE(TestKernels);

XPU_EXPORT(test_constants);

XPU_EXPORT(get_driver_type);
int get_driver_type::operator()(xpu::driver_t *driver) {
    #if XPU_IS_CPU
        *driver = xpu::cpu;
    #elif XPU_IS_HIP
        *driver = xpu::hip;
    #elif XPU_IS_CUDA
        *driver = xpu::cuda;
    #endif
    return 0;
}

XPU_D void do_vector_add(const float *x, const float *y, float *z, int N) {
    int iThread = xpu::block_idx::x() * xpu::block_dim::x() + xpu::thread_idx::x();
    if (iThread >= N) {
        return;
    }
    z[iThread] = x[iThread] + y[iThread];
}

// Ensure that kernels without arguments can compile.
XPU_EXPORT(empty_kernel);
XPU_D void empty_kernel::operator()(context &) {}

XPU_EXPORT(vector_add);
XPU_D void vector_add::operator()(context &, const float *x, const float *y, float *z, int N) {
    do_vector_add(x, y, z, N);
}

XPU_EXPORT(vector_add_timing0);
XPU_D void vector_add_timing0::operator()(context &, const float *x, const float *y, float *z, int N) {
    do_vector_add(x, y, z, N);
}

XPU_EXPORT(vector_add_timing1);
XPU_D void vector_add_timing1::operator()(context &, const float *x, const float *y, float *z, int N) {
    do_vector_add(x, y, z, N);
}

XPU_EXPORT(sort_float);
XPU_D void sort_float::operator()(context &ctx, float *items, int N, float *buf, float **dst) {
    float *res = sort_t{ctx.smem()}.sort(items, N, buf, [](const float &x) { return x; });

    if (xpu::block_idx::x() == 0) {
        *dst = res;
    }
}

XPU_EXPORT(sort_struct);
XPU_D void sort_struct::operator()(context &ctx, key_value_t *items, int N, key_value_t *buf, key_value_t **dst) {
    key_value_t *res = sort_t{ctx.smem()}.sort(items, N, buf, [](const key_value_t &pair) { return pair.key; });

    if (xpu::block_idx::x() == 0) {
        *dst = res;
    }
}

XPU_EXPORT(merge);
XPU_D void merge::operator()(context &ctx, const float *a, size_t size_a, const float *b, size_t size_b, float *dst) {
    merge_t(ctx.smem()).merge(a, size_a, b, size_b, dst, [](float a, float b) { return a < b; });
}

XPU_EXPORT(merge_single);
XPU_D void merge_single::operator()(context &ctx, const float *a, size_t size_a, const float *b, size_t size_b, float *dst) {
    merge_t(ctx.smem()).merge(a, size_a, b, size_b, dst, [](float a, float b) { return a < b; });
}

XPU_EXPORT(block_scan);
XPU_D void block_scan::operator()(context &ctx, int *incl, int *excl) {
    scan_t scan{ctx.smem()};
    scan.inclusive_sum(1, incl[xpu::thread_idx::x()]);
    scan.exclusive_sum(1, excl[xpu::thread_idx::x()]);
}

XPU_EXPORT(access_cmem);
XPU_D void access_cmem::operator()(context &, float3_ *out) {
    if (xpu::thread_idx::x() > 0) {
        return;
    }
    const float3_ &in = xpu::cmem<test_constants>();
    *out = in;
}

XPU_D void get_thread_idx(int *thread_idx, int *block_dim, int *block_idx, int *grid_dim) {
    int threadsPerBlock = xpu::block_dim::x() * xpu::block_dim::y() * xpu::block_dim::z();
    int threadIdxInBlock = xpu::block_dim::x() * xpu::block_dim::y() * xpu::thread_idx::z() + xpu::block_dim::x() * xpu::thread_idx::y() + xpu::thread_idx::x();
    int blockNumInGrid = xpu::grid_dim::x() * xpu::grid_dim::y() * xpu::block_idx::z() + xpu::grid_dim::x() * xpu::block_idx::y() + xpu::block_idx::x();

    int iThread = blockNumInGrid * threadsPerBlock + threadIdxInBlock;

    thread_idx[iThread * 3 + 0] = xpu::thread_idx::x();
    thread_idx[iThread * 3 + 1] = xpu::thread_idx::y();
    thread_idx[iThread * 3 + 2] = xpu::thread_idx::z();
    block_dim[iThread * 3 + 0] = xpu::block_dim::x();
    block_dim[iThread * 3 + 1] = xpu::block_dim::y();
    block_dim[iThread * 3 + 2] = xpu::block_dim::z();
    block_idx[iThread * 3 + 0] = xpu::block_idx::x();
    block_idx[iThread * 3 + 1] = xpu::block_idx::y();
    block_idx[iThread * 3 + 2] = xpu::block_idx::z();
    grid_dim[iThread * 3 + 0] = xpu::grid_dim::x();
    grid_dim[iThread * 3 + 1] = xpu::grid_dim::y();
    grid_dim[iThread * 3 + 2] = xpu::grid_dim::z();
}

XPU_EXPORT(get_thread_idx_1d);
XPU_D void get_thread_idx_1d::operator()(context &, int *thread_idx, int *block_dim, int *block_idx, int *grid_dim) {
    get_thread_idx(thread_idx, block_dim, block_idx, grid_dim);
}

XPU_EXPORT(get_thread_idx_2d);
XPU_D void get_thread_idx_2d::operator()(context &, int *thread_idx, int *block_dim, int *block_idx, int *grid_dim) {
    get_thread_idx(thread_idx, block_dim, block_idx, grid_dim);
}

XPU_EXPORT(get_thread_idx_3d);
XPU_D void get_thread_idx_3d::operator()(context &, int *thread_idx, int *block_dim, int *block_idx, int *grid_dim) {
    get_thread_idx(thread_idx, block_dim, block_idx, grid_dim);
}

XPU_EXPORT(test_device_funcs);
XPU_D void test_device_funcs::operator()(context &, variant *out) {
    if (xpu::thread_idx::x() > 0) {
        return;
    }

    out[ABS].f = xpu::abs(-1.f);
    out[ACOS].f = xpu::acos(-1.f);
    out[ACOSH].f = xpu::acosh(1.f);
    out[ASIN].f = xpu::asin(1.f);
    out[ASINH].f = xpu::asinh(1.f);
    out[ATAN2].f = xpu::atan2(1.f, 1.f);
    out[ATAN].f = xpu::atan(1.f);
    out[ATANH].f = xpu::atanh(0.9f);
    out[CBRT].f = xpu::cbrt(729.f);
    out[CEIL].f = xpu::ceil(2.4f);
    out[COPYSIGN].f = xpu::copysign(1.f, -2.f);
    out[COS].f = xpu::cos(xpu::pi() / 3.f);
    out[COSH].f = xpu::cosh(1.f);
    out[COSPI].f = xpu::cospi(xpu::pi() / 3.f);
    out[ERF].f = xpu::erf(1.f);
    out[ERFC].f = xpu::erfc(0.f);
    out[EXP2].f = xpu::exp2(4.f);
    out[EXP].f = xpu::exp(2.f);
    out[EXPM1].f = xpu::expm1(1.f);
    out[FDIM].f = xpu::fdim(4, 1);
    out[FLOOR].f = xpu::floor(2.6f);
    out[FMA].f = xpu::fma(2.f, 3.f, 4.f);
    out[FMOD].f = xpu::fmod(5.1f, 3.f);
    out[HYPOT].f = xpu::hypot(1.f, 1.f);
    out[ILOGB].i = xpu::ilogb(123.45);
    out[ISFINITE].b = xpu::isfinite(xpu::pi());
    out[ISINF].b = xpu::isinf(INFINITY);
    out[ISNAN].b = xpu::isnan(xpu::nan(""));
    out[J0].f = xpu::j0(1.f);
    out[J1].f = xpu::j1(1.f);
    out[JN].f = xpu::jn(2, 1.f);
    out[LDEXP].f = xpu::ldexp(7.f, -4);
    out[LLRINT].ll = xpu::llrint(2.5f);
    out[LLROUND].ll = xpu::llround(2.4f);
    out[LOG].f = xpu::log(1.f);
    out[LOG10].f = xpu::log10(1000.f);
    out[LOG1P].f = xpu::log1p(0.f);
    out[LOG2].f = xpu::log2(32.f);
    out[LOGB].f = xpu::logb(123.45f);
    out[LRINT].ll = xpu::lrint(2.5f);
    out[LROUND].ll = xpu::lround(2.4f);
    out[MAX].f = xpu::max(-1.f, 1.f);
    out[MIN].f = xpu::min(-1.f, 1.f);
    out[NEARBYINT].f = xpu::nearbyint(2.3f);
    float a[5] = {1.f, 1.f, 1.f, 1.f, 1.f};
    out[NORM].f = xpu::norm(5, a);
    out[NORM3D].f = xpu::norm3d(2.f, 3.f, 4.f);
    out[NORM4D].f = xpu::norm4d(2.f, 3.f, 4.f, 5.f);
    out[POW].f = xpu::pow(3.f, 3.f);
    out[RCBRT].f = xpu::rcbrt(27.f);
    out[REMAINDER].f = xpu::remainder(5.1, 3.0);
    out[REMQUO_REM].f = xpu::remquo(10.3f, 4.5f, &out[REMQUO_QUO].i);
    out[RHYPOT].f = xpu::rhypot(2.f, 3.f);
    out[RINT].f = xpu::rint(2.4f);
    out[RNORM].f = xpu::rnorm(5, a);
    out[RNORM3D].f = xpu::rnorm3d(2.f, 3.f, 4.f);
    out[RNORM4D].f = xpu::rnorm4d(2.f, 3.f, 4.f, 5.f);
    out[ROUND].f = xpu::round(2.6f);
    out[RSQRT].f = xpu::rsqrt(4.f);
    out[SCALBLN].f = xpu::scalbln(7.f, -4);
    out[SCALBN].f = xpu::scalbn(7.f, -4);
    out[SIGNBIT].b = xpu::signbit(-3.f);
    xpu::sincos(xpu::pi(), &out[SINCOS_SIN].f , &out[SINCOS_COS].f);
    xpu::sincospi(1.f, &out[SINCOSPI_SIN].f , &out[SINCOSPI_COS].f);
    out[SIN].f = xpu::sin(xpu::pi());
    out[SINH].f = xpu::sinh(1.f);
    out[SINPI].f = xpu::sinpi(1.f);
    out[SQRT].f = xpu::sqrt(64.f);
    out[TAN].f = xpu::tan(xpu::pi_4());
    out[TANH].f = xpu::tanh(1.f);
    out[TGAMMA].f = xpu::tgamma(10);
    out[TRUNC].f = xpu::trunc(2.7f);
    out[Y0].f = xpu::y0(1.f);
    out[Y1].f = xpu::y1(1.f);
    out[YN].f = xpu::yn(2, 1.f);
}
