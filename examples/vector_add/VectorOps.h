#pragma once

#include <xpu/device.h>

struct VectorOps {};

struct VectorAdd : xpu::kernel<VectorOps> {
    using context = xpu::kernel_context<xpu::no_smem>;
    XPU_D void operator()(context &, const float *, const float *, float *, size_t);
};
