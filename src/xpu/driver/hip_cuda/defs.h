#ifndef XPU_DRIVER_CUDA_DEFS_H
#define XPU_DRIVER_CUDA_DEFS_H

#define HIP_ENABLE_PRINTF

#if !defined(__NVCC__) && !defined(__HIPCC__)
#error "Target is not Cuda or Hip."
#endif

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

#define XPU_IS_CPU 0
#ifdef __NVCC__
#define XPU_IS_CUDA 1
#define XPU_IS_HIP 0
#else
#define XPU_IS_CUDA 0
#define XPU_IS_HIP 1
#endif

#define XPU_D __device__

#define XPU_FORCE_INLINE __forceinline__

#define XPU_NO_INLINE __noinline__

#define XPU_UNITY_BUILD 1

#endif
