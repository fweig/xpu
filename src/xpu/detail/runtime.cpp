#include "runtime.h"
#include "../host.h"

#include <cstdlib>
#include <sstream>

#define CATCH_ERROR(err) CATCH_ERROR_D(active_driver_type, err)
#define CATCH_ERROR_D(driver, err) throw_on_driver_error(driver, err)

using namespace xpu::detail;

runtime &runtime::instance() {
    static runtime the_runtime{};
    return the_runtime;
}

void runtime::initialize(driver target_driver) {

    int target_device = 0;

    const char *use_logger = std::getenv("XPU_VERBOSE");
    bool verbose = (use_logger != nullptr) && (std::string{use_logger} != "0");

    if (verbose) {
        logger::instance().initialize([](const char *msg) {
            std::cerr << msg << std::endl;
        });
    }

    const char *profile_env = std::getenv("XPU_PROFILE");
    measure_time = (profile_env != nullptr) && (std::string{profile_env} != "0");

    const char *device_env = std::getenv("XPU_DEVICE");
    if (device_env != nullptr) {
        std::vector<std::pair<std::string, xpu::driver>> str_to_driver {
            {"cpu", driver::cpu},
            {"cuda", driver::cuda},
            {"hip", driver::hip},
        };

        bool valid_driver = false;
        for (auto &driver_str : str_to_driver) {
            const std::string &name = driver_str.first;
            if (strncmp(device_env, name.c_str(), name.size()) == 0) {
                valid_driver = true;
                device_env += name.size();
                target_driver = driver_str.second;
                break;
            }
        }

        if (not valid_driver) {
            throw exception{"Requested unknown driver with environment variable XPU_DEVICE: " + std::string{device_env}};
        }

        sscanf(device_env, "%d", &target_device);
    }

    this->the_cpu_driver.reset(new cpu_driver{});
    error err = 0;
    switch (target_driver) {
    case driver::cpu:
        break;
    case driver::cuda:
        the_cuda_driver.reset(new lib_obj<driver_interface>{"libxpu_Cuda.so"});
        break;
    case driver::hip:
        the_hip_driver.reset(new lib_obj<driver_interface>{"libxpu_Hip.so"});
        break;
    }
    active_driver_type = target_driver;
    err = get_active_driver()->setup(target_device);
    CATCH_ERROR(err);
    if (active_driver_type != driver::cpu) {
        the_cpu_driver->setup(0);
    }
    XPU_LOG("Set %s as active driver.", driver_str(target_driver));
}

void *runtime::host_malloc(size_t bytes) {
    void *ptr = nullptr;
    error err = the_cpu_driver->device_malloc(&ptr, bytes);

    CATCH_ERROR(err);

    return ptr;
}

void *runtime::device_malloc(size_t bytes) {
    void *ptr = nullptr;
    error err = get_active_driver()->device_malloc(&ptr, bytes);
    CATCH_ERROR(err);
    return ptr;
}

void runtime::free(void *ptr) {
    error err = get_active_driver()->free(ptr);
    CATCH_ERROR(err);
}

void runtime::memcpy(void *dst, const void *src, size_t bytes) {
    error err = get_active_driver()->memcpy(dst, src, bytes);
    CATCH_ERROR(err);
}

void runtime::memset(void *dst, int ch, size_t bytes) {
    error err = get_active_driver()->memset(dst, ch, bytes);
    CATCH_ERROR(err);
}

driver_interface *runtime::get_driver(driver d) const {
    switch (d) {
        case driver::cpu: return the_cpu_driver.get();
        case driver::cuda: return the_cuda_driver->obj;
        case driver::hip: return the_hip_driver->obj;
    }

    return nullptr; // UNREACHABLE;
}

driver_interface *runtime::get_active_driver() const {
    return get_driver(active_driver_type);
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

void runtime::throw_on_driver_error(driver d, error err) const {
    if (err == 0) {
        return;
    }

    std::stringstream ss;
    ss << "xpu: Driver " << driver_str(d) <<  " raised error " << err << ": " << get_driver(d)->error_to_string(err);

    throw exception(ss.str());
}
