#include "cuda_driver.h"

xpu::error cuda_driver::setup() {
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    return 0;
}

xpu::error cuda_driver::device_malloc(void **ptr, size_t bytes) {
    return cudaMalloc(ptr, bytes);
}

xpu::error cuda_driver::free(void *ptr) {
    return cudaFree(ptr);
}

xpu::error cuda_driver::memcpy(void *dst, const void *src, size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
}

extern "C" xpu::driver_interface *create() {
    return new cuda_driver{};
}

extern "C" void destroy(xpu::driver_interface *b) {
    delete b;
}