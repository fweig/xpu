#pragma once

#include <xpu/device.h>

struct VectorOps {};

XPU_EXPORT_KERNEL(VectorOps, VectorAdd, const float *, const float * , float *, size_t);
