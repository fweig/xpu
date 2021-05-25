#include "../../detail/driver_interface.h"

#include <hip/hip_runtime_api.h>

#include <iostream>

using xpu::detail::error;

class hip_driver : public xpu::detail::driver_interface {

public:
    error setup() override {
        error err;

        hipDeviceProp_t props;
        err = hipGetDeviceProperties(&props, 0);
        if (err != 0) {
            return err;
        }

        std::cout << "xpu: selected hip device " << props.name << "(" << props.major << props.minor << ")" << std::endl;

        err = hipSetDevice(0);
        if (err != 0) {
            return err;
        }
        err = hipDeviceSynchronize();

        return err;
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

};

extern "C" xpu::detail::driver_interface *create() {
    return new hip_driver{};
}

extern "C" void destroy(xpu::detail::driver_interface *b) {
    delete b;
}
