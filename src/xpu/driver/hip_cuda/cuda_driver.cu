#include "../../detail/driver_interface.h"
#include "../../detail/log.h"
#include "../../common.h"

#include <iostream>

namespace xpu {
namespace detail {

class cuda_driver : public driver_interface {

public:
    virtual ~cuda_driver() {}

    error setup() override {
        return 0;
    }

    error device_malloc(void **ptr, size_t bytes) override {
        return cudaMalloc(ptr, bytes);
    }

    error free(void *ptr) override {
        return cudaFree(ptr);
    }

    error memcpy(void *dst, const void *src, size_t bytes) override {
        error err = cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
        cudaDeviceSynchronize();
        return err;
    }

    error memset(void *dst, int ch, size_t bytes) override {
        return cudaMemset(dst, ch, bytes);
    }

    error num_devices(int *devices) override {
        return cudaGetDeviceCount(devices);
    }

    error set_device(int device) override {
        return cudaSetDevice(device);
    }

    error get_device(int *device) override {
        return cudaGetDevice(device);
    }

    error device_synchronize() override {
        return cudaDeviceSynchronize();
    }

    error get_properties(device_prop *props, int device) override {
        cudaDeviceProp cuprop;
        error err = cudaGetDeviceProperties(&cuprop, device);
        if (err != 0) {
            return err;
        }

        props->name = cuprop.name;
        props->driver_type = xpu::driver::cuda;
        props->major = cuprop.major;
        props->minor = cuprop.minor;

        return 0;
    }

    error meminfo(size_t *free, size_t *total) override {
        return cudaMemGetInfo(free, total);
    }

    const char *error_to_string(error err) override {
        return cudaGetErrorString(static_cast<cudaError_t>(err));
    }

};

} // namespace detail
} // namespace xpu

extern "C" xpu::detail::driver_interface *create() {
    return new xpu::detail::cuda_driver{};
}

extern "C" void destroy(xpu::detail::driver_interface *b) {
    delete b;
}
