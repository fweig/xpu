#ifndef XPU_DRIVER_CUDA_DEFS_H
#define XPU_DRIVER_CUDA_DEFS_H

#define XPU_IS_CPU 0
#define XPU_IS_CUDA 1
#define XPU_D __device__
#define XPU_INLINE __forceinline__
#define XPU_NO_INLINE __noinline__
#define XPU_UNITY_BUILD 1
#define XPU_DRIVER_NAME CUDA

#endif