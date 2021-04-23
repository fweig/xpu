#include "runtime.h"

using namespace xpu::detail;

static runtime &runtime::instance() {
    runtime the_runtime{};
    return the_runtime;
}

void runtime::initialize(driver t) {
    this->cpu_driver = std::unique_ptr<driver_interface>(new cpu_driver{});
    this->cpu_driver->setup();
    error err = 0;
    switch (t) {
    case driver::cpu:
        active_driver_inst = cpu_driver.get();
        std::cout << "xpu: set cpu as active driver" << std::endl;
        break;
    case driver::cuda:
        std::cout << "xpu: try to setup cuda driver" << std::endl;
        cuda_driver.reset(new lib_obj<driver_interface>{"libxpu_driver_Cuda.so"});
        err = cuda_driver->obj->setup();
        if (err != 0) {
            throw exception{"Caught error " + std::to_string(err)};
        }
        active_driver_inst = cuda_driver->obj;
        std::cout << "xpu: set cuda as active driver" << std::endl;
        break;
    case driver::hip:
        std::cout << "xpu: try to setup hip driver" << std::endl;
        hip_driver.reset(new lib_obj<driver_interface>{"libxpu_driver_Hip.so"});
        err = hip_driver->obj->setup();
        if (err != 0) {
            throw exception{"Caught error " + std::to_string(err)};
        }
        active_driver_inst = hip_driver->obj;
        std::cout << "xpu: set hip as active driver" << std::endl;
        break;
    }
    active_driver_type = t;
}

void *runtime::device_malloc(size_t bytes) {
    void *ptr = nullptr;
    error err = active_backend_inst->device_malloc(&ptr, bytes);

    if (err != 0) {
        throw exception{"Caught error " + std::to_string(err)};
    }

    return ptr;
}

void runtime::free(void *ptr) {
    error err = active_backend_inst->free(ptr);
    if (err != 0) {
        throw exception{"Caught error " + std::to_string(err)};
    }
}

void runtime::memcpy(void *dst, const void *src, size_t bytes) {
    error err = active_backend_inst->memcpy(dst, src, bytes);
    if (err != 0) {
        throw exception{"Caught error " + std::to_string(err)};
    }
}

void runtime::memset(void *dst, int ch, size_t bytes) {
    error err = active_backend_inst->memcpy(dst, ch, bytes);
    if (err != 0) {
        throw exception{"Caught error " + std::to_string(err)};
    }
}
