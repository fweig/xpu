#ifndef XPU_DETAIL_DEFINES_H
#define XPU_DETAIL_DEFINES_H

#ifndef XPU_DEFINES_H
#error "This header should not be included directly. Use xpu/defines.h instead."
#endif

#ifdef __NVCC__
#define XPU_DETAIL_IS_CUDA 1
#define XPU_DETAIL_IS_HIP 0
#define XPU_DETAIL_IS_CPU 0
#define XPU_DETAIL_COMPILATION_TARGET (xpu::driver::cuda)
#elif defined(__HIPCC__)
#define XPU_DETAIL_IS_CUDA 0
#define XPU_DETAIL_IS_HIP 1
#define XPU_DETAIL_IS_CPU 0
#define XPU_DETAIL_COMPILATION_TARGET (xpu::driver::hip)
#else
#define XPU_DETAIL_IS_CUDA 0
#define XPU_DETAIL_IS_HIP 0
#define XPU_DETAIL_IS_CPU 1
#define XPU_DETAIL_COMPILATION_TARGET (xpu::driver::cpu)
#endif

#if XPU_IS_CUDA || XPU_IS_HIP
#define XPU_DETAIL_FORCE_INLINE __forceinline__
#define XPU_DETAIL_DEVICE_SPEC __device__
#else
#define XPU_DETAIL_FORCE_INLINE inline __attribute__((always_inline))
#define XPU_DETAIL_DEVICE_SPEC
#endif

#if XPU_IS_HIP
#include <hip/hip_runtime.h>
#endif

#endif
