#include "../../detail/driver_interface.h"
#include "../../detail/log.h"

#include <iostream>

#define CU_DEVICE 0

namespace xpu {
namespace detail {

class cuda_driver : public driver_interface {

public:
    virtual ~cuda_driver() {}

    error setup() override {
        error err;
        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, CU_DEVICE);
        if (err != 0) {
            return err;
        }

        XPU_LOG("Selected %s(arch = %d%d) as active device.", props.name, props.major, props.minor);

        err = cudaSetDevice(CU_DEVICE);
        if (err != 0) {
            return err;
        }
        err = cudaDeviceSynchronize();

        return err;
    }

    error device_malloc(void **ptr, size_t bytes) override {
        size_t free, total;
        cudaMemGetInfo(&free, &total);

        XPU_LOG("Allocating %lu bytes on active CUDA device. (%lu / %lu free)", bytes, free, total);

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

};

} // namespace detail
} // namespace xpu

extern "C" xpu::detail::driver_interface *create() {
    return new xpu::detail::cuda_driver{};
}

extern "C" void destroy(xpu::detail::driver_interface *b) {
    delete b;
}
