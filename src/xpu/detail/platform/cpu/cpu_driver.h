#ifndef XPU_DRIVER_CPU_CPU_DRIVER_H
#define XPU_DRIVER_CPU_CPU_DRIVER_H

#include "../../backend_base.h"

namespace xpu::detail {

class cpu_driver : public backend_base {

public:
    virtual ~cpu_driver() {}

    error setup() override;
    error malloc_device(void **, size_t) override;
    error malloc_host(void **, size_t) override;
    error malloc_shared(void **, size_t) override;
    error free(void *) override;

    error create_queue(void **, int) override;
    error destroy_queue(void *) override;
    error synchronize_queue(void *) override;

    error memcpy_async(void *, const void *, size_t, void *, double *) override;
    error memset_async(void *, int, size_t, void *, double *) override;

    error num_devices(int *) override;
    error set_device(int) override;
    error get_device(int *) override;
    error get_properties(device_prop *, int) override;
    error get_ptr_prop(const void *, int *, mem_type *) override;

    error meminfo(size_t *, size_t *) override;

    const char *error_to_string(error) override;

    driver_t get_type() override;

private:
    enum error_code : int {
        SUCCESS = 0,
        OUT_OF_MEMORY,
        INVALID_DEVICE,
        MACOSX_ERROR
    };

};

} // namespace xpu::detail

#endif
