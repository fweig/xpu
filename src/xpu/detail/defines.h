#ifndef XPU_DETAIL_DEFINES_H
#define XPU_DETAIL_DEFINES_H

#ifndef XPU_DEFINES_H
#error "This header should not be included directly. Use xpu/defines.h instead."
#endif

#ifdef __NVCC__
#define XPU_DETAIL_IS_CUDA 1
#define XPU_DETAIL_IS_HIP 0
#define XPU_DETAIL_IS_SYCL 0
#define XPU_DETAIL_IS_CPU 0
#define XPU_DETAIL_COMPILATION_TARGET (xpu::cuda)
#elif defined(__HIPCC__)
#define XPU_DETAIL_IS_CUDA 0
#define XPU_DETAIL_IS_HIP 1
#define XPU_DETAIL_IS_SYCL 0
#define XPU_DETAIL_IS_CPU 0
#define XPU_DETAIL_COMPILATION_TARGET (xpu::hip)
#elif defined(SYCL_LANGUAGE_VERSION)
#define XPU_DETAIL_IS_CUDA 0
#define XPU_DETAIL_IS_HIP 0
#define XPU_DETAIL_IS_SYCL 1
#define XPU_DETAIL_IS_CPU 0
#define XPU_DETAIL_COMPILATION_TARGET (xpu::sycl)
#else
#define XPU_DETAIL_IS_CUDA 0
#define XPU_DETAIL_IS_HIP 0
#define XPU_DETAIL_IS_SYCL 0
#define XPU_DETAIL_IS_CPU 1
#define XPU_DETAIL_COMPILATION_TARGET (xpu::cpu)
#endif

#if XPU_IS_CUDA || XPU_IS_HIP
#define XPU_DETAIL_FORCE_INLINE __forceinline__
#define XPU_DETAIL_DEVICE_SPEC __device__
#define XPU_DETAIL_HOST_SPEC __host__
#define XPU_DETAIL_CONSTANT_SPEC __constant__
#else
#define XPU_DETAIL_FORCE_INLINE inline __attribute__((always_inline))
#define XPU_DETAIL_DEVICE_SPEC
#define XPU_DETAIL_HOST_SPEC
#define XPU_DETAIL_CONSTANT_SPEC
#endif

#if XPU_IS_CUDA
    #ifdef __CUDA_ARCH__
        #define XPU_DETAIL_IS_DEVICE_CODE 1
    #else
        #define XPU_DETAIL_IS_DEVICE_CODE 0
    #endif
#elif XPU_IS_HIP
    #ifdef __HIP_DEVICE_COMPILE__
        #define XPU_DETAIL_IS_DEVICE_CODE 1
    #else
        #define XPU_DETAIL_IS_DEVICE_CODE 0
    #endif
#elif XPU_IS_SYCL
    #ifdef __SYCL_DEVICE_ONLY__
        #define XPU_DETAIL_IS_DEVICE_CODE 1
    #else
        #define XPU_DETAIL_IS_DEVICE_CODE 0
    #endif
#else // XPU_IS_CPU
    #define XPU_DETAIL_IS_DEVICE_CODE 0
#endif

#if XPU_IS_CUDA
#if __CUDA_ARCH__ >= 600
#define XPU_DETAIL_CUDA_HAS_BLOCK_ATOMICS 1
#else
#define XPU_DETAIL_CUDA_HAS_BLOCK_ATOMICS 0
#endif
#else
#define XPU_DETAIL_CUDA_HAS_BLOCK_ATOMICS 0
#endif

#if XPU_IS_HIP
#include <hip/hip_runtime.h>
#endif

#endif
