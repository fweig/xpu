#ifndef XPU_DETAIL_CONSTANTS_H
#define XPU_DETAIL_CONSTANTS_H

#define _USE_MATH_DEFINES
#include <cmath>

#ifdef M_PIf32
#define F32_CONSTANT(x) x ## f32
#else
#define F32_CONSTANT(x) float(x)
#endif

namespace xpu {
XPU_D constexpr float pi() { return F32_CONSTANT(M_PI); }
XPU_D constexpr float pi_2() { return F32_CONSTANT(M_PI_2); }
XPU_D constexpr float pi_4() { return F32_CONSTANT(M_PI_4); }
XPU_D constexpr float deg_to_rad() { return pi() / 180.f; }
XPU_D constexpr float sqrt2() { return F32_CONSTANT(M_SQRT2); }
} // namespace xpu

#undef F32_CONSTANT

#endif
