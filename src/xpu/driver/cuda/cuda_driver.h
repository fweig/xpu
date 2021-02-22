#ifndef XPU_DRIVER_CUDA_CUDA_DRIVER_H
#define XPU_DRIVER_CUDA_CUDA_DRIVER_H

#include "../../host.h"

class cuda_driver : public xpu::driver_interface {

public:
    xpu::error setup() override;
    xpu::error device_malloc(void **, size_t) override;
    xpu::error free(void *) override;
    xpu::error memcpy(void *, const void *, size_t) override;
    xpu::error memset(void *, int, size_t) override;
};

#endif