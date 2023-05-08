#ifndef XPU_DETAIL_BACKEND_BASE_H
#define XPU_DETAIL_BACKEND_BASE_H

#include "common.h"

#include <cstddef>

namespace xpu::detail {

class backend_base {

public:
    virtual ~backend_base() {}

    virtual error setup() = 0;
    virtual error malloc_device(void **, size_t) = 0;
    virtual error malloc_host(void **, size_t) = 0;
    virtual error malloc_shared(void **, size_t) = 0;
    virtual error free(void *) = 0;

    virtual error create_queue(void **, int) = 0;
    virtual error destroy_queue(void *) = 0;
    virtual error synchronize_queue(void *) = 0;

    virtual error memcpy_async(void *, const void *, size_t, void *, double *) = 0;
    virtual error memset_async(void *, int, size_t, void *, double *) = 0;

    virtual error num_devices(int *) = 0;
    virtual error set_device(int) = 0;
    virtual error get_device(int *) = 0;
    virtual error get_properties(device_prop *, int) = 0;
    virtual error get_ptr_prop(const void *, int *, mem_type *) = 0;

    virtual error meminfo(size_t *, size_t *) = 0;

    virtual const char *error_to_string(error) = 0;

    virtual driver_t get_type() = 0;
};

} // namespace xpu::detail

#endif
