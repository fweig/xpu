#ifndef XPU_DEFS_H
#define XPU_DEFS_H

#ifdef __NVCC__
#include "driver/cuda/defs.h"
#else // CPU
#include "driver/cpu/defs.h"
#endif

#ifndef XPU_IS_CPU
#error "XPU_IS_CPU is not defined."
#endif

#ifndef XPU_IS_CUDA
#error "XPU_IS_CUDA is not defined."
#endif

#ifndef XPU_D
#error "XPU_D not defined."
#endif

#ifndef XPU_INLINE
#error "XPU_INLINE not defined."
#endif

#ifndef XPU_NO_INLINE
#error "XPU_NO_LINE not defined."
#endif

#ifndef XPU_UNITY_BUILD
#error "XPU_UNITY_BUILD not defined."
#endif

#endif