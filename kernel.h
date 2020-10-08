#pragma once

#include "gpu.h"

#ifdef __NVCC__
#include "backend/cuda/kernel.h"
#else // CPU
#include "backend/cpu/kernel.h"
#endif