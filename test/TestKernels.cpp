#include "TestKernels.h"

XPU_IMAGE(TestKernels);

XPU_EXPORT(test_constant0);
XPU_EXPORT(test_constant1);
XPU_EXPORT(test_constant2);
XPU_EXPORT(cmem_buffer);

XPU_EXPORT(get_driver_type);
int get_driver_type::operator()(xpu::driver_t *driver) {
    #if XPU_IS_CPU
        *driver = xpu::cpu;
    #elif XPU_IS_HIP
        *driver = xpu::hip;
    #elif XPU_IS_CUDA
        *driver = xpu::cuda;
    #elif XPU_IS_SYCL
        *driver = xpu::sycl;
    #else
        #error "Unknown driver"
    #endif
    return 0;
}

XPU_D void do_vector_add(xpu::tpos &pos, const float *x, const float *y, float *z, size_t N) {
    xpu::view<const float> xv{x, N};
    xpu::view<const float> yv{y, N};
    xpu::view<float> zv{z, N};
    size_t iThread = pos.block_idx_x() * pos.block_dim_x() + pos.thread_idx_x();
    if (iThread >= xv.size()) {
        return;
    }
    zv[iThread] = xv[iThread] + yv[iThread];
}

// Ensure that kernels without arguments can compile.
XPU_EXPORT(empty_kernel);
XPU_D void empty_kernel::operator()(context &) {}

XPU_EXPORT(buffer_access);
XPU_D void buffer_access::operator()(context &ctx, xpu::buffer<int> buf) {
    int iThread = ctx.pos().block_idx_x() * ctx.pos().block_dim_x() + ctx.pos().thread_idx_x();
    if (iThread > 0) {
        return;
    }
    // printf("buf = %p\n", buf.get());
    *buf = 42;
}

XPU_EXPORT(vector_add);
XPU_D void vector_add::operator()(context &ctx, const float *x, const float *y, float *z, int N) {
    do_vector_add(ctx.pos(), x, y, z, static_cast<size_t>(N));
}

XPU_EXPORT(vector_add_timing0);
XPU_D void vector_add_timing0::operator()(context &ctx, const float *x, const float *y, float *z, int N) {
    do_vector_add(ctx.pos(), x, y, z, static_cast<size_t>(N));
}

XPU_EXPORT(vector_add_timing1);
XPU_D void vector_add_timing1::operator()(context &ctx, const float *x, const float *y, float *z, int N) {
    do_vector_add(ctx.pos(), x, y, z, static_cast<size_t>(N));
}

XPU_EXPORT(sort_float);
XPU_D void sort_float::operator()(context &ctx, float *items, int N, float *buf, float **dst) {
#ifndef DONT_TEST_BLOCK_SORT
    float *res = sort_t{ctx.pos(), ctx.smem()}.sort(items, N, buf, [](const float &x) { return x; });

    if (ctx.pos().block_idx_x() == 0) {
        *dst = res;
    }
#endif
}

XPU_EXPORT(sort_struct);
XPU_D void sort_struct::operator()(context &ctx, key_value_t *items, int N, key_value_t *buf, key_value_t **dst) {
#ifndef DONT_TEST_BLOCK_SORT
    key_value_t *res = sort_t{ctx.pos(), ctx.smem()}.sort(items, N, buf, [](const key_value_t &pair) { return pair.key; });

    if (ctx.pos().block_idx_x() == 0) {
        *dst = res;
    }
#endif
}

XPU_EXPORT(merge);
XPU_D void merge::operator()(context &ctx, const float *a, size_t size_a, const float *b, size_t size_b, float *dst) {
#ifndef DONT_TEST_BLOCK_SORT
    merge_t(ctx.pos(), ctx.smem()).merge(a, size_a, b, size_b, dst, [](float a, float b) { return a < b; });
#endif
}

XPU_EXPORT(merge_single);
XPU_D void merge_single::operator()(context &ctx, const float *a, size_t size_a, const float *b, size_t size_b, float *dst) {
#ifndef DONT_TEST_BLOCK_SORT
    merge_t(ctx.pos(), ctx.smem()).merge(a, size_a, b, size_b, dst, [](float a, float b) { return a < b; });
#endif
}

XPU_EXPORT(block_scan);
XPU_D void block_scan::operator()(context &ctx, int *incl, int *excl) {
#ifndef DONT_TEST_BLOCK_FUNCS
    scan_t scan{ctx.pos(), ctx.smem()};
    xpu::tpos &pos = ctx.pos();
    scan.inclusive_sum(1, incl[pos.thread_idx_x()]);
    scan.exclusive_sum(1, excl[pos.thread_idx_x()]);
#endif
}

XPU_EXPORT(access_cmem_single);
XPU_D void access_cmem_single::operator()(context &ctx, float3_ *out) {
    if (ctx.pos().thread_idx_x() > 0) {
        return;
    }
    const float3_ &in = ctx.cmem().get<test_constant0>();
    *out = in;
}

XPU_EXPORT(access_cmem_multiple);
XPU_D void access_cmem_multiple::operator()(context &ctx, float3_ *out0, double *out1, float *out2) {
    if (ctx.pos().thread_idx_x() > 0) {
        return;
    }
    const float3_ &in0 = ctx.cmem().get<test_constant0>();
    *out0 = in0;
    const double &in1 = ctx.cmem().get<test_constant1>();
    *out1 = in1;
    const float &in2 = ctx.cmem().get<test_constant2>();
    *out2 = in2;
}

XPU_D void get_thread_idx(xpu::tpos &pos, int *thread_idx, int *block_dim, int *block_idx, int *grid_dim) {
    int threadsPerBlock = pos.block_dim_x() * pos.block_dim_y() * pos.block_dim_z();
    int threadIdxInBlock = pos.block_dim_x() * pos.block_dim_y() * pos.thread_idx_z() + pos.block_dim_x() * pos.thread_idx_y() + pos.thread_idx_x();
    int blockNumInGrid = pos.grid_dim_x() * pos.grid_dim_y() * pos.block_idx_z() + pos.grid_dim_x() * pos.block_idx_y() + pos.block_idx_x();

    int iThread = blockNumInGrid * threadsPerBlock + threadIdxInBlock;

    thread_idx[iThread * 3 + 0] = pos.thread_idx_x();
    thread_idx[iThread * 3 + 1] = pos.thread_idx_y();
    thread_idx[iThread * 3 + 2] = pos.thread_idx_z();
    block_dim[iThread * 3 + 0] = pos.block_dim_x();
    block_dim[iThread * 3 + 1] = pos.block_dim_y();
    block_dim[iThread * 3 + 2] = pos.block_dim_z();
    block_idx[iThread * 3 + 0] = pos.block_idx_x();
    block_idx[iThread * 3 + 1] = pos.block_idx_y();
    block_idx[iThread * 3 + 2] = pos.block_idx_z();
    grid_dim[iThread * 3 + 0] = pos.grid_dim_x();
    grid_dim[iThread * 3 + 1] = pos.grid_dim_y();
    grid_dim[iThread * 3 + 2] = pos.grid_dim_z();
}

XPU_EXPORT(get_thread_idx_1d);
XPU_D void get_thread_idx_1d::operator()(context &ctx, int *thread_idx, int *block_dim, int *block_idx, int *grid_dim) {
    get_thread_idx(ctx.pos(), thread_idx, block_dim, block_idx, grid_dim);
}

XPU_EXPORT(get_thread_idx_2d);
XPU_D void get_thread_idx_2d::operator()(context &ctx, int *thread_idx, int *block_dim, int *block_idx, int *grid_dim) {
    get_thread_idx(ctx.pos(), thread_idx, block_dim, block_idx, grid_dim);
}

XPU_EXPORT(get_thread_idx_3d);
XPU_D void get_thread_idx_3d::operator()(context &ctx, int *thread_idx, int *block_dim, int *block_idx, int *grid_dim) {
    get_thread_idx(ctx.pos(), thread_idx, block_dim, block_idx, grid_dim);
}

XPU_EXPORT(test_device_funcs);
XPU_D void test_device_funcs::operator()(context &ctx, variant *out) {
    if (ctx.pos().thread_idx_x() > 0) {
        return;
    }

    out[ABS].f = xpu::abs(-1.f);
    out[ACOS].f = xpu::acos(-1.f);
    out[ACOSH].f = xpu::acosh(1.f);
    out[ACOSPI].f = xpu::acospi(-1.f);
    out[ASIN].f = xpu::asin(1.f);
    out[ASINH].f = xpu::asinh(1.f);
    out[ASINPI].f = xpu::asinpi(1.f);
    out[ATAN2].f = xpu::atan2(1.f, 1.f);
    out[ATAN].f = xpu::atan(1.f);
    out[ATANH].f = xpu::atanh(0.9f);
    out[ATANPI].f = xpu::atanpi(1.f);
    out[ATAN2PI].f = xpu::atan2pi(1.f, 1.f);
    out[CBRT].f = xpu::cbrt(729.f);
    out[CEIL].f = xpu::ceil(2.4f);
    out[COPYSIGN].f = xpu::copysign(1.f, -2.f);
    out[COS].f = xpu::cos(xpu::pi() / 3.f);
    out[COSH].f = xpu::cosh(1.f);
    out[COSPI].f = xpu::cospi(xpu::pi() / 3.f);
    out[ERF].f = xpu::erf(1.f);
    out[ERFC].f = xpu::erfc(0.f);
    out[EXP2].f = xpu::exp2(4.f);
    out[EXP10].f = xpu::exp10(4.f);
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
    out[NORM3D].f = xpu::norm3d(2.f, 3.f, 4.f);
    out[NORM4D].f = xpu::norm4d(2.f, 3.f, 4.f, 5.f);
    out[POW].f = xpu::pow(3.f, 3.f);
    out[RCBRT].f = xpu::rcbrt(27.f);
    out[REMAINDER].f = xpu::remainder(5.1, 3.0);
    out[REMQUO_REM].f = xpu::remquo(10.3f, 4.5f, &out[REMQUO_QUO].i);
    out[RHYPOT].f = xpu::rhypot(2.f, 3.f);
    out[RINT].f = xpu::rint(2.4f);
    out[RNORM3D].f = xpu::rnorm3d(2.f, 3.f, 4.f);
    out[RNORM4D].f = xpu::rnorm4d(2.f, 3.f, 4.f, 5.f);
    out[ROUND].f = xpu::round(2.6f);
    out[RSQRT].f = xpu::rsqrt(4.f);
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
}

XPU_EXPORT(templated_kernel<0>);
XPU_EXPORT(templated_kernel<1>);
XPU_EXPORT(templated_kernel<42>);
template<int N>
XPU_D void templated_kernel<N>::operator()(context &, int *out) {
    *out = N;
}
