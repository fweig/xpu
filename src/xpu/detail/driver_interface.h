#ifndef XPU_DETAIL_DRIVER_INTERFACE_H
#define XPU_DETAIL_DRIVER_INTERFACE_H

#include "common.h"
#include "../common.h"

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

    virtual error num_devices(int *) = 0;
    virtual error set_device(int) = 0;
    virtual error get_device(int *) = 0;
    virtual error device_synchronize() = 0;
    virtual error get_properties(device_prop *, int) = 0;

    virtual error meminfo(size_t *, size_t *) = 0;

    virtual const char *error_to_string(error) = 0;

    virtual driver_t get_type() = 0;

};

} // namespace detail
} // namespace xpu

#endif
