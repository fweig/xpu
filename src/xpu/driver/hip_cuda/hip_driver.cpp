#include "../../detail/driver_interface.h"
#include "../../detail/log.h"

#include <hip/hip_runtime_api.h>

#include <iostream>

namespace xpu {
namespace detail {

class hip_driver : public driver_interface {

public:
    virtual ~hip_driver() {}

    error setup(int device) override {
        error err;

        hipDeviceProp_t props;
        err = hipGetDeviceProperties(&props, device);
        if (err != 0) {
            return err;
        }

        XPU_LOG("Selected %s(arch = %d%d) as active device.", props.name, props.major, props.minor);

        err = hipSetDevice(device);
        if (err != 0) {
            return err;
        }
        err = hipDeviceSynchronize();

        return err;
    }

    error device_malloc(void **ptr, size_t bytes) override {
        size_t free, total;
        hipMemGetInfo(&free, &total);

        XPU_LOG("Allocating %lu bytes on active HIP device. (%lu / %lu free)", bytes, free, total);

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
