#pragma once

#include "xpu.h"

#include <cstddef>

namespace xpu {

class driver_interface {

public:
    virtual xpu::error setup() = 0;
    virtual xpu::error device_malloc(void **, size_t) = 0;
    virtual xpu::error free(void *) = 0;
    virtual xpu::error memcpy(void *, const void *, size_t) = 0;

};

}

// vim: set ts=4 sw=4 sts=4 expandtab: