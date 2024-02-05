#ifndef XPU_DETAIL_PLATFORM_HIP_CUDA_PRELUDE_H
#define XPU_DETAIL_PLATFORM_HIP_CUDA_PRELUDE_H

#include "../../macros.h"
#include "../../../defines.h"

#if XPU_IS_CUDA
    #define CUHIP(expr) XPU_CONCAT(cuda, expr)
    #define XPU_HIP_VERSION_AT_LEAST(major, minor) (0)
    using cuhip_device_prop = cudaDeviceProp;
    using cuhip_pointer_attributes = cudaPointerAttributes;
#else
    #include <hip/hip_runtime_api.h>
    #define CUHIP(expr) XPU_CONCAT(hip, expr)

    #define XPU_HIP_VERSION (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR)
    #define XPU_HIP_VERSION_AT_LEAST(major, minor) (XPU_HIP_VERSION >= (major * 100 + minor))

    #if XPU_HIP_VERSION_AT_LEAST(6, 0)
        #define HIP_PTR_TYPE(ptrattr) (ptrattr).type
    #else // HIP_VERSION < 6.0
        #define HIP_PTR_TYPE(ptrattr) (ptrattr).memoryType
    #endif

    using cuhip_device_prop = hipDeviceProp_t;
    using cuhip_pointer_attributes = hipPointerAttribute_t;
#endif

#endif // XPU_DETAIL_PLATFORM_HIP_CUDA_PRELUDE_H
