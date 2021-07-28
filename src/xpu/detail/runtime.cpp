#include "runtime.h"
#include "../host.h"

#include <cstdlib>
#include <sstream>

#define DRIVER_CALL(func) throw_on_driver_error(active_driver(), get_active_driver()->func)
#define CPU_DRIVER_CALL(func) throw_on_driver_error(xpu::cpu, the_cpu_driver->func)

using namespace xpu::detail;

runtime &runtime::instance() {
    static runtime the_runtime{};
    return the_runtime;
}

void runtime::initialize(driver_t target_driver) {

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
        std::vector<std::pair<std::string, xpu::driver_t>> str_to_driver {
            {"cpu", cpu},
            {"cuda", cuda},
            {"hip", hip},
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
    switch (target_driver) {
    case cpu:
        break;
    case cuda:
        XPU_LOG("Loading cuda driver.");
        the_cuda_driver.reset(new lib_obj<driver_interface>{"libxpu_Cuda.so"});
        if (not the_cuda_driver->ok()) {
            throw exception{"xpu: Requested cuda driver, but failed to load 'libxpu_Cuda.so'."};
        }
        break;
    case hip:
        XPU_LOG("Loading hip driver.");
        the_hip_driver.reset(new lib_obj<driver_interface>{"libxpu_Hip.so"});
        if (not the_hip_driver->ok()) {
            throw exception{"xpu: Requested hip driver, but failed to load 'libxpu_Hip.so'."};
        }
        break;
    }
    _active_driver = target_driver;

    DRIVER_CALL(setup());

    device_prop props;
    DRIVER_CALL(get_properties(&props, target_device));
    DRIVER_CALL(set_device(target_device));
    DRIVER_CALL(device_synchronize());

    if (_active_driver != cpu) {
        XPU_LOG("Selected %s(arch = %d%d) as active device.", props.name.c_str(), props.major, props.minor);
        CPU_DRIVER_CALL(setup());
    } else {
        XPU_LOG("Selected %s as active device.", props.name.c_str());
    }}

void *runtime::host_malloc(size_t bytes) {
    void *ptr = nullptr;
    CPU_DRIVER_CALL(device_malloc(&ptr, bytes));
    return ptr;
}

void *runtime::device_malloc(size_t bytes) {
    if (logger::instance().active()) {
        size_t free, total;
        DRIVER_CALL(meminfo(&free, &total));
        int device;
        DRIVER_CALL(get_device(&device));
        device_prop props;
        DRIVER_CALL(get_properties(&props, device));
        XPU_LOG("Allocating %lu bytes on device %s. [%lu / %lu available]", bytes, props.name.c_str(), free, total);
    }
    void *ptr = nullptr;
    DRIVER_CALL(device_malloc(&ptr, bytes));
    return ptr;
}

void runtime::free(void *ptr) {
    DRIVER_CALL(free(ptr));
}

void runtime::memcpy(void *dst, const void *src, size_t bytes) {
    DRIVER_CALL(memcpy(dst, src, bytes));
}

void runtime::memset(void *dst, int ch, size_t bytes) {
    DRIVER_CALL(memset(dst, ch, bytes));
}

driver_interface *runtime::get_driver(driver_t d) const {
    switch (d) {
        case cpu: return the_cpu_driver.get();
        case cuda: return the_cuda_driver->obj;
        case hip: return the_hip_driver->obj;
    }

    return nullptr; // UNREACHABLE;
}

driver_interface *runtime::get_active_driver() const {
    return get_driver(_active_driver);
}

std::string runtime::complete_file_name(const char *fname, driver_t d) const {
    std::string prefix = "lib";
    std::string suffix = "";
    switch (d) {
    case cuda: suffix = "_Cuda.so"; break;
    case hip:  suffix = "_Hip.so"; break;
    case cpu:  suffix = ".so"; break;
    }
    return prefix + std::string{fname} + suffix;
}

const char *runtime::driver_str(driver_t d) const {
    switch (d) {
    case cpu: return "CPU";
    case cuda: return "CUDA";
    case hip: return "HIP";
    }
    return "UNKNOWN";
}

void runtime::throw_on_driver_error(driver_t d, error err) const {
    if (err == 0) {
        return;
    }

    std::stringstream ss;
    ss << "xpu: Driver " << driver_str(d) <<  " raised error " << err << ": " << get_driver(d)->error_to_string(err);

    throw exception(ss.str());
}
