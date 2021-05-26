#ifndef XPU_DETAIL_DRIVER_INTERFACE_H
#define XPU_DETAIL_DRIVER_INTERFACE_H

#include "common.h"

#include <cstddef>

namespace xpu {
namespace detail {

class driver_interface {

public:
    virtual ~driver_interface() {}

    virtual error setup() = 0;
    virtual error device_malloc(void **, size_t) = 0;
    virtual error free(void *) = 0;
    virtual error memcpy(void *, const void *, size_t) = 0;
    virtual error memset(void *, int, size_t) = 0;

};

}
}

#endif
