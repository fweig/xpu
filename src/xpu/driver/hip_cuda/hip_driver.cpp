#include "../../detail/driver_interface.h"
#include "../../detail/log.h"
#include "../../common.h"

#include <hip/hip_runtime_api.h>

#include <iostream>

namespace xpu {
namespace detail {

class hip_driver : public driver_interface {

public:
    virtual ~hip_driver() {}

    error setup() override {
        return 0;
    }

    error device_malloc(void **ptr, size_t bytes) override {
        return hipMalloc(ptr, bytes);
    }

    error free(void *ptr) override {
        return hipFree(ptr);
    }

    error memcpy(void *dst, const void *src, size_t bytes) override {
        error err = hipMemcpy(dst, src, bytes, hipMemcpyDefault);
        hipDeviceSynchronize();
        return err;
    }

    error memset(void *dst, int ch, size_t bytes) override {
        return hipMemset(dst, ch, bytes);
    }

    error num_devices(int *devices) override {
        return hipGetDeviceCount(devices);
    }

    error set_device(int device) override {
        return hipSetDevice(device);
    }

    error get_device(int *device) override {
        return hipGetDevice(device);
    }

    error device_synchronize() override {
        return hipDeviceSynchronize();
    }

    error get_properties(device_prop *props, int device) override {
        hipDeviceProp_t cuprop;
        error err = hipGetDeviceProperties(&cuprop, device);
        if (err != 0) {
            return err;
        }

        props->name = cuprop.name;
        props->driver = hip;
        props->major = cuprop.major;
        props->minor = cuprop.minor;

        return 0;
    }

    error meminfo(size_t *free, size_t *total) override {
        return hipMemGetInfo(free, total);
    }

    const char *error_to_string(error err) override {
        return hipGetErrorString(static_cast<hipError_t>(err));
    }

};

} // namespace detail
} // namespace xpu

extern "C" xpu::detail::driver_interface *create() {
    return new xpu::detail::hip_driver{};
}

extern "C" void destroy(xpu::detail::driver_interface *b) {
    delete b;
}
