#include "gpu.h"

#include "GPUBackend.h"

// TODO: load from dyn lib
#include "backend/cpu/CPUBackend.h"

#include <memory>

static std::unique_ptr<GPUBackend> theBackend;

namespace gpu {

    void initialize(GPUBackendType t) {
        theBackend = std::unique_ptr<GPUBackend>(new CPUBackend{});
        theBackend->setup();
    }

    void *malloc(size_t bytes) {
        void *ptr;
        // TODO: check for errors
        theBackend->deviceMalloc(&ptr, bytes);
        return ptr;
    }

    void free(void *ptr) {
        // TODO: implement
    }

    void memcpy(void *dst, const void *src, size_t bytes) {
        // TODO: check for errors
        theBackend->memcpy(dst, src, bytes);
    }

}

