#pragma once

#include "defs.h"
#include "host.h"

namespace xpu {
    struct no_smem {};
}

#if defined(__NVCC__)
#include "driver/cuda/device_runtime.h"
#else // CPU
#include "driver/cpu/device_runtime.h"
#endif

#ifndef XPU_KERNEL
#error "XPU_KERNEL not defined."
#endif