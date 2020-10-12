#pragma once

#include "xpu.h"

#if defined(__NVCC__)
#include "driver/cuda/kernel.h"
#else // CPU
#include "driver/cpu/kernel.h"
#endif