#pragma once

#include "gpu.h"

#if defined(__NVCC__)
#include "backend/cuda/kernel.h"
#else // CPU
#include "backend/cpu/kernel.h"
#endif