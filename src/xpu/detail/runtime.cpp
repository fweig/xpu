#include "runtime.h"
#include "../host.h"

using namespace xpu::detail;

runtime &runtime::instance() {
    static runtime the_runtime{};
    return the_runtime;
}

void runtime::initialize(driver t) {
    this->the_cpu_driver = std::unique_ptr<cpu_driver>(new cpu_driver{});
    this->the_cpu_driver->setup();
    error err = 0;
    switch (t) {
    case driver::cpu:
        active_driver_inst = the_cpu_driver.get();
        std::cout << "xpu: set cpu as active driver" << std::endl;
        break;
    case driver::cuda:
        std::cout << "xpu: try to setup cuda driver" << std::endl;
        the_cuda_driver.reset(new lib_obj<driver_interface>{"libxpu_Cuda.so"});
        err = the_cuda_driver->obj->setup();
        if (err != 0) {
            throw exception{"Caught error " + std::to_string(err)};
        }
        active_driver_inst = the_cuda_driver->obj;
        std::cout << "xpu: set cuda as active driver" << std::endl;
        break;
    case driver::hip:
        std::cout << "xpu: try to setup hip driver" << std::endl;
        the_hip_driver.reset(new lib_obj<driver_interface>{"libxpu_Hip.so"});
        err = the_hip_driver->obj->setup();
        if (err != 0) {
            throw exception{"Caught error " + std::to_string(err)};
        }
        active_driver_inst = the_hip_driver->obj;
        std::cout << "xpu: set hip as active driver" << std::endl;
        break;
    }
    active_driver_type = t;
}

void *runtime::host_malloc(size_t bytes) {
    void *ptr = nullptr;
    error err = the_cpu_driver->device_malloc(&ptr, bytes);

    if (err != 0) {
        throw exception{"Caught error " + std::to_string(err)};
    }

    return ptr;
}

void *runtime::device_malloc(size_t bytes) {
    void *ptr = nullptr;
    error err = active_driver_inst->device_malloc(&ptr, bytes);

    if (err != 0) {
        throw exception{"Caught error " + std::to_string(err)};
    }

    return ptr;
}

void runtime::free(void *ptr) {
    error err = active_driver_inst->free(ptr);
    if (err != 0) {
        throw exception{"Caught error " + std::to_string(err)};
    }
}

void runtime::memcpy(void *dst, const void *src, size_t bytes) {
    error err = active_driver_inst->memcpy(dst, src, bytes);
    if (err != 0) {
        throw exception{"Caught error " + std::to_string(err)};
    }
}

void runtime::memset(void *dst, int ch, size_t bytes) {
    error err = active_driver_inst->memset(dst, ch, bytes);
    if (err != 0) {
        throw exception{"Caught error " + std::to_string(err)};
    }
}

std::string runtime::complete_file_name(const char *fname, driver d) const {
    std::string prefix = "lib";
    std::string suffix = "";
    switch (d) {
    case xpu::driver::cuda: suffix = "_Cuda.so"; break;
    case xpu::driver::hip:  suffix = "_Hip.so"; break;
    case xpu::driver::cpu:  suffix = ".so"; break;
    }
    return prefix + std::string{fname} + suffix;
}
