#pragma once

#include "../../driver_interface.h"

class cuda_driver : public xpu::driver_interface {

public:
    xpu::error setup() override;
    xpu::error device_malloc(void **, size_t) override;
    xpu::error free(void *) override;
    xpu::error memcpy(void *, const void *, size_t) override;
};