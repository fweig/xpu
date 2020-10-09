#include "gpu.h"

#include "GPUBackend.h"
#include "dl_utils.h"
#include "backend/cpu/CPUBackend.h"

#include <memory>

namespace gpu {
    class BackendLoader;
}

static std::unique_ptr<GPUBackend> theCPUBackend;
static std::unique_ptr<LibObj<GPUBackend>> theCUDABackend;
static GPUBackend *activeBackendInst = nullptr;

static GPUBackendType activeBackendType; 

namespace gpu {
    void initialize(GPUBackendType t) {
        theCPUBackend = std::unique_ptr<GPUBackend>(new CPUBackend{});
        theCPUBackend->setup();
        switch (t){
        case GPUBackendType::CPU:
            activeBackendInst = theCPUBackend.get();
            break;
        case GPUBackendType::CUDA:       
            theCUDABackend.reset(new LibObj<GPUBackend>("./build/libXPUBackendCUDA.so"));
            theCUDABackend->obj->setup();
            activeBackendInst = theCUDABackend->obj;
        }
         activeBackendType = t;
    }

    void *malloc(size_t bytes) {
        void *ptr = nullptr;
        // TODO: check for errors
        activeBackendInst->deviceMalloc(&ptr, bytes);
        return ptr;
    }

    void free(void *ptr) {
        // TODO: check for errors
        activeBackendInst->free(ptr);
    }

    void memcpy(void *dst, const void *src, size_t bytes) {
        // TODO: check for errors
        activeBackendInst->memcpy(dst, src, bytes);
    }

    GPUBackendType activeBackend() {
        return activeBackendType;
    }


}

