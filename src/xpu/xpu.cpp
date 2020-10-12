#include "xpu.h"

#include "driver_interface.h"
#include "dl_utils.h"
#include "driver/cpu/cpu_driver.h"

#include <memory>

static std::unique_ptr<xpu::driver_interface> theCPUBackend;
static std::unique_ptr<LibObj<xpu::driver_interface>> theCUDABackend;
static xpu::driver_interface *activeBackendInst = nullptr;

static xpu::driver activeBackendType; 

namespace xpu {
    void initialize(xpu::driver t) {
        theCPUBackend = std::unique_ptr<driver_interface>(new cpu_driver{});
        theCPUBackend->setup();
        switch (t){
        case driver::cpu:
            activeBackendInst = theCPUBackend.get();
            break;
        case driver::cuda:       
            theCUDABackend.reset(new LibObj<driver_interface>("./build/libXPUBackendCUDA.so"));
            theCUDABackend->obj->setup();
            activeBackendInst = theCUDABackend->obj;
        }
         activeBackendType = t;
    }

    void *device_malloc(size_t bytes) {
        void *ptr = nullptr;
        // TODO: check for errors
        activeBackendInst->device_malloc(&ptr, bytes);
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

    driver active_driver() {
        return activeBackendType;
    }
}

