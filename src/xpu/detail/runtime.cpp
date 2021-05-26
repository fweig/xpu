#include "runtime.h"
#include "../host.h"

#include <cstdlib>

using namespace xpu::detail;

runtime &runtime::instance() {
    static runtime the_runtime{};
    return the_runtime;
}

void runtime::initialize(driver t) {
    const char *use_logger = std::getenv("XPU_VERBOSE");
    bool verbose = (use_logger != nullptr) && (std::string{use_logger} != "0");

    if (verbose) {
        logger::instance().initialize([](const char *msg) {
            std::cerr << msg << std::endl;
        });
    }

    this->the_cpu_driver.reset(new cpu_driver{});
    this->the_cpu_driver->setup();
    error err = 0;
    switch (t) {
    case driver::cpu:
        active_driver_inst = the_cpu_driver.get();
        break;
    case driver::cuda:
        the_cuda_driver.reset(new lib_obj<driver_interface>{"libxpu_Cuda.so"});
        err = the_cuda_driver->obj->setup();
        if (err != 0) {
            throw exception{"Caught error " + std::to_string(err)};
        }
        active_driver_inst = the_cuda_driver->obj;
        break;
    case driver::hip:
        the_hip_driver.reset(new lib_obj<driver_interface>{"libxpu_Hip.so"});
        err = the_hip_driver->obj->setup();
        if (err != 0) {
            throw exception{"Caught error " + std::to_string(err)};
        }
        active_driver_inst = the_hip_driver->obj;
        break;
    }
    active_driver_type = t;
    XPU_LOG("Set %s as active driver.", driver_str(t));
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

const char *runtime::driver_str(driver d) const {
    switch (d) {
    case driver::cpu: return "CPU";
    case driver::cuda: return "CUDA";
    case driver::hip: return "HIP";
    }
    return "UNKNOWN";
}
