#pragma once

#ifndef GPU_DEVICE_KEYWORD
#error("")
#endif

#ifndef GPU_INLINE_KEYWORD
#error("")
#endif

#ifndef GPU_HOST_KEYWORD
#error("")
#endif

#define GPU_D GPU_DEVICE_KEYWORD
#define GPU_DI GPU_DEVICE_KEYWORD GPU_INLINE_KEYWORD
#define GPU_H GPU_HOST_KEYWORD
#define GPU_DH GPU_DEVICE_KEYWORD GPU_HOST_KEYWORD
