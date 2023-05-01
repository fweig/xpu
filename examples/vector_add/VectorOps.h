#pragma once

#include <xpu/device.h>

struct VectorOps : xpu::device_image {};

struct VectorAdd : xpu::kernel<VectorOps> {
    using context = xpu::kernel_context<xpu::no_smem>;
    XPU_D void operator()(context &, xpu::buffer<const float>, xpu::buffer<const float>, xpu::buffer<float>, size_t);
};
