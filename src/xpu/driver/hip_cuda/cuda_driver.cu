#include "../../detail/driver_interface.h"

#include <iostream>

#define CU_DEVICE 0

using xpu::detail::error;

class cuda_driver : public xpu::detail::driver_interface {

public:
    error setup() override {
        std::cout << "xpu: CUDA SETUP" << std::endl;

        error err;
        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, CU_DEVICE);
        if (err != 0) {
            return err;
        }

        std::cout << "xpu: selected cuda device " << props.name << "(" << props.major << props.minor << ")" << std::endl;

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
        std::cout << "cuda driver: free memory = " << free << "; total = " << total << std::endl;
        std::cout << "cuda driver: allocating " << bytes << " bytes" << std::endl;
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

extern "C" xpu::detail::driver_interface *create() {
    return new cuda_driver{};
}

extern "C" void destroy(xpu::detail::driver_interface *b) {
    delete b;
}
