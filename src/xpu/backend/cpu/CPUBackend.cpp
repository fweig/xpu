#include "CPUBackend.h"

#include <cstdlib>
#include <cstring>

GPUError CPUBackend::setup() {
    return 0;
}

GPUError CPUBackend::deviceMalloc(void ** ptr, size_t bytes) {
    *ptr = std::malloc(bytes); 
    return 0;
}

GPUError CPUBackend::free(void *ptr) {
    std::free(ptr);
}

GPUError CPUBackend::memcpy(void *dst, const void *src, size_t bytes) {
    std::memcpy(dst, src, bytes);
    return 0;
}

extern "C" {
    GPUBackend *createBackend() {
        return new CPUBackend{};
    }
}