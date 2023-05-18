/**
 * @file defines.h
 *
 * @brief Defines for xpu.
 *
 * Include as `#include <xpu/defines.h>`.
 */
#ifndef XPU_DEFINES_H
#define XPU_DEFINES_H

#define XPU_IS_CPU XPU_DETAIL_IS_CPU
#define XPU_IS_CUDA XPU_DETAIL_IS_CUDA
#define XPU_IS_HIP XPU_DETAIL_IS_HIP
#define XPU_IS_SYCL XPU_DETAIL_IS_SYCL

#define XPU_IS_HIP_CUDA (XPU_IS_CUDA || XPU_IS_HIP)

/**
 * @brief Function specifier for device functions. (Replaces __device__)
 */
#define XPU_D XPU_DETAIL_DEVICE_SPEC

/**
 * @brief Function specifier for host functions. (Replaces __host__)
 * @note This is usually not required, as all device functions are also
 *   compiled for the host by xpu. You only need XPU_H if the function
 *   is called from host code _inside_ the device image. This happens
 *   typically if:
 *     - The funcions is called within a xpu::function object.
 *     - For copy constructors and assignment operators of types that are
 *       passed by value to kernels.
 */
#define XPU_H XPU_DETAIL_HOST_SPEC

/**
 * @brief Is true if the current compilation target is a device.
 */
#define XPU_IS_DEVICE_CODE XPU_DETAIL_IS_DEVICE_CODE

#define XPU_FORCE_INLINE XPU_DETAIL_FORCE_INLINE

#define XPU_COMPILATION_TARGET XPU_DETAIL_COMPILATION_TARGET

#define XPU_CUDA_HAS_BLOCK_ATOMICS XPU_DETAIL_HAS_BLOCK_ATOMICS

#include "detail/defines.h"

#endif
