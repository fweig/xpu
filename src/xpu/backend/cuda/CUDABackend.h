#pragma once

#include "../../GPUBackend.h"

class CUDABackend : public GPUBackend {

public:
    GPUError setup() override;
    GPUError deviceMalloc(void **, size_t) override;
    GPUError free(void *) override;
    GPUError memcpy(void *, const void *, size_t) override;
};