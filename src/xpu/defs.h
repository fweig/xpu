#pragma once

#if defined(__NVCC__)
#include "driver/cuda/defs.h"
#else // CPU
#include "driver/cpu/defs.h"
#endif