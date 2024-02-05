#ifndef XPU_DRIVER_SYCL_SYCL_DRIVER_H
#define XPU_DRIVER_SYCL_SYCL_DRIVER_H

#include "../../backend_base.h"

#include <sycl/sycl.hpp>

#include <memory>

namespace xpu::detail {

class sycl_driver : public backend_base {

public:
    virtual ~sycl_driver() {}

    sycl::queue get_queue(void *);

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
    sycl::property_list m_prop_list;
    sycl::device m_device_obj;
    sycl::context m_context;
    // sycl::queue m_default_queue;
    int m_device = -1;

    std::vector<std::unique_ptr<sycl::queue>> m_queues;

    int get_device_id(sycl::device);
};

} // namespace xpu::detail

#endif
