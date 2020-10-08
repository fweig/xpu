#pragma once

#include <cstddef>

using GPUError = int;

class GPUBackend {

public:
    virtual GPUError setup() = 0;
    virtual GPUError deviceMalloc(void **, size_t) = 0;
    virtual GPUError memcpy(void *, const void *, size_t) = 0;

};

// vim: set ts=4 sw=4 sts=4 expandtab: