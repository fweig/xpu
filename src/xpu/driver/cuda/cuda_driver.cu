#include "cuda_driver.h"

#include <iostream>

xpu::error cuda_driver::setup() {
    std::cout << "xpu: CUDA SETUP" << std::endl;

    xpu::error err;
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, 0);
    if (err != 0) {
        return err;
    }

    std::cout << "xpu: selected cuda device " << props.name << "(" << props.major << props.minor << ")" << std::endl;

    err = cudaSetDevice(0);
    if (err != 0) {
        return err;
    }
    err = cudaDeviceSynchronize();


    return err;
}

xpu::error cuda_driver::device_malloc(void **ptr, size_t bytes) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "cuda driver: free memory = " << free << "; total = " << total << std::endl;
    std::cout << "cuda driver: allocating " << bytes << " bytes" << std::endl;
    return cudaMalloc(ptr, bytes);
}

xpu::error cuda_driver::free(void *ptr) {
    return cudaFree(ptr);
}

xpu::error cuda_driver::memcpy(void *dst, const void *src, size_t bytes) {
    xpu::error err = cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
    cudaDeviceSynchronize();
    return err;
}

xpu::error cuda_driver::memset(void *dst, int ch, size_t bytes) {
    return cudaMemset(dst, ch, bytes);
}

extern "C" xpu::driver_interface *create() {
    return new cuda_driver{};
}

extern "C" void destroy(xpu::driver_interface *b) {
    delete b;
}