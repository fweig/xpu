#ifndef XPU_DRIVER_INTERFACE_H
#define XPU_DRIVER_INTERFACE_H

#include "internals.h"

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

#endif