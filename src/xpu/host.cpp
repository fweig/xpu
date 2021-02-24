#include "host.h"
#include "driver/cpu/cpu_driver.h"

#include <memory>

static std::unique_ptr<xpu::driver_interface> theCPUBackend;
static std::unique_ptr<xpu::lib_obj<xpu::driver_interface>> theCUDABackend;
static std::unique_ptr<xpu::lib_obj<xpu::driver_interface>> theHIPBackend;
static xpu::driver_interface *activeBackendInst = nullptr;

static xpu::driver activeBackendType; 

namespace xpu {
    void initialize(xpu::driver t) {
        theCPUBackend = std::unique_ptr<driver_interface>(new cpu_driver{});
        theCPUBackend->setup();
        xpu::error err = 0;
        switch (t) {
        case driver::cpu:
            activeBackendInst = theCPUBackend.get();
            std::cout << "xpu: set cpu as active driver" << std::endl;
            break;
        case driver::cuda:
            std::cout << "xpu: try to setup cuda driver" << std::endl;
            theCUDABackend.reset(new lib_obj<driver_interface>("libxpu_driver_Cuda.so"));
            err = theCUDABackend->obj->setup();
            if (err != 0) {
                throw exception{"Caught error " + std::to_string(err)};
            }
            activeBackendInst = theCUDABackend->obj;
            std::cout << "xpu: set cuda as active driver" << std::endl;
            break;
        case driver::hip:
            std::cout << "xpu: try to setup hip driver" << std::endl;
            theHIPBackend.reset(new lib_obj<driver_interface>("libxpu_driver_Hip.so"));
            err = theHIPBackend->obj->setup();
            if (err != 0) {
                throw exception{"Caught error " + std::to_string(err)};
            }
            activeBackendInst = theHIPBackend->obj;
            std::cout << "xpu: set hip as active driver" << std::endl;
            break;
        }
        activeBackendType = t;
    }

    void *host_malloc(size_t bytes) {
        void *ptr = nullptr;
        error err = theCPUBackend->device_malloc(&ptr, bytes);
        if (err != 0) {
            throw exception{"Caught error " + err};
        }

        return ptr;
    }

    void *device_malloc(size_t bytes) {
        void *ptr = nullptr;
        // TODO: check for errors
        error err = activeBackendInst->device_malloc(&ptr, bytes);
        if (err != 0) {
            throw exception{"Caught error " + std::to_string(err)};
        }

        return ptr;
    }

    void free(void *ptr) {
        error err = activeBackendInst->free(ptr);
        if (err != 0) {
            throw exception{"Caught error " + std::to_string(err)};
        }
    }

    void memcpy(void *dst, const void *src, size_t bytes) {
        error err = activeBackendInst->memcpy(dst, src, bytes);
        if (err != 0) {
            throw exception{"Caught error " + std::to_string(err)};
        }
    }

    void memset(void *dst, int ch, size_t bytes) {
        error err = activeBackendInst->memset(dst, ch, bytes);
        if (err != 0) {
            throw exception{"Caught error " + std::to_string(err)};
        }
    }

    driver active_driver() {
        return activeBackendType;
    }
}

