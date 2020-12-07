#pragma once

#include "defs.h"
#include "internals.h"

namespace xpu {
    struct no_smem {};
    struct kernel_info {
    dim i_thread;
    dim n_threads;
    dim i_block;
    dim n_blocks;
};
}

#if defined(__NVCC__)
#include "driver/cuda/device_runtime.h"
#else // CPU
#include "driver/cpu/device_runtime.h"
#endif

#ifndef XPU_KERNEL
#error "XPU_KERNEL not defined."
#endif