#include "CUDABackend.h"

GPUError CUDABackend::setup() {
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    return 0;
}

GPUError CUDABackend::deviceMalloc(void **ptr, size_t bytes) {
    return cudaMalloc(ptr, bytes);
}

GPUError CUDABackend::free(void *ptr) {
    return cudaFree(ptr);
}

GPUError CUDABackend::memcpy(void *dst, const void *src, size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
}

extern "C" GPUBackend *create() {
    return new CUDABackend{};
}

extern "C" void destroy(GPUBackend *b) {
    delete b;
}