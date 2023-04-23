#ifndef XPU_DETAIL_PLATFORM_HIP_CUDA_PRELUDE_H
#define XPU_DETAIL_PLATFORM_HIP_CUDA_PRELUDE_H

#include "../../macros.h"
#include "../../../defines.h"

#if XPU_IS_CUDA
    #define CUHIP(expr) XPU_CONCAT(cuda, expr)
    using cuhip_device_prop = cudaDeviceProp;
    using cuhip_pointer_attributes = cudaPointerAttributes;
#else
    #include <hip/hip_runtime_api.h>
    #define CUHIP(expr) XPU_CONCAT(hip, expr)
    using cuhip_device_prop = hipDeviceProp_t;
    using cuhip_pointer_attributes = hipPointerAttribute_t;
#endif

#endif // XPU_DETAIL_PLATFORM_HIP_CUDA_PRELUDE_H
