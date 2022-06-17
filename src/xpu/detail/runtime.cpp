#include "runtime.h"
#include "../host.h"

#include <cstdlib>
#include <sstream>

#define DRIVER_CALL_I(type, func) throw_on_driver_error((type), get_driver((type))->func)
#define DRIVER_CALL(func) DRIVER_CALL_I(active_driver(), func)
#define CPU_DRIVER_CALL(func) DRIVER_CALL_I(cpu, func)

using namespace xpu::detail;

bool runtime::getenv_bool(std::string name, bool fallback) {
    const char *env = getenv(name.c_str());
    return (env == nullptr ? fallback : (strcmp(env, "0") != 0));
}

std::string runtime::getenv_str(std::string name, std::string_view fallback) {
    const char *env = getenv(name.c_str());
    return (env == nullptr ? std::string{fallback} : std::string{env});
}

runtime &runtime::instance() {
    static runtime the_runtime{};
    return the_runtime;
}

void runtime::initialize() {

    bool verbose = getenv_bool("XPU_VERBOSE", false);
    if (verbose) {
        logger::instance().initialize([](const char *msg) {
            std::cerr << msg << std::endl;
        });
    }

    m_measure_time = getenv_bool("XPU_PROFILE", false);

    int target_device = 0;
    driver_t target_driver = cpu;
    if (auto device_env = getenv_str("XPU_DEVICE", "cpu"); device_env != "cpu") {
        std::vector<std::pair<std::string, xpu::driver_t>> str_to_driver {
            {"cpu", cpu},
            {"cuda", cuda},
            {"hip", hip},
        };

        bool valid_driver = false;
        for (auto &driver_str : str_to_driver) {
            const std::string &name = driver_str.first;
            if (strncmp(device_env.c_str(), name.c_str(), name.size()) == 0) {
                valid_driver = true;
                device_env += name.size();
                target_driver = driver_str.second;
                break;
            }
        }

        if (not valid_driver) {
            throw exception{"Requested unknown driver with environment variable XPU_DEVICE: " + device_env};
        }

        sscanf(device_env.c_str(), "%d", &target_device);
    }

    m_cpu_driver.reset(new cpu_driver{});
    CPU_DRIVER_CALL(setup());

    XPU_LOG("Loading cuda driver.");
    m_cuda_driver.reset(new lib_obj<driver_interface>{"libxpu_Cuda.so"});
    if (m_cuda_driver->ok()) {
        DRIVER_CALL_I(cuda, setup());
        XPU_LOG("Finished loading cuda driver.");
    } else {
        XPU_LOG("Couldn't find 'libxpu_Cuda.so'. Cuda driver not active.");
    }

    XPU_LOG("Loading hip driver.");
    m_hip_driver.reset(new lib_obj<driver_interface>{"libxpu_Hip.so"});
    if (m_hip_driver->ok()) {
        DRIVER_CALL_I(hip, setup());
        XPU_LOG("Finished loading hip driver.");
    } else {
        XPU_LOG("Couldn't find 'libxpu_Hip.so'. Hip driver not active.");
    }

    XPU_LOG("Found devices:");
    for (driver_t driver : {cpu, cuda, hip}) {
        if (not has_driver(driver)) {
            continue;
        }
        int ndevices;
        DRIVER_CALL_I(driver, num_devices(&ndevices));
        for (int i = 0; i < ndevices; i++) {
            device_prop props;
            DRIVER_CALL_I(driver, get_properties(&props, i));
            if (props.driver != cpu) {
                XPU_LOG("  %lu: %s (arch = %d%d)", m_devices.size(), props.name.c_str(), props.major, props.minor);
            } else {
                XPU_LOG("  %lu: %s", m_devices.size(), props.name.c_str());
            }
            m_devices.push_back(props);
            m_devices_by_driver[props.driver].push_back(props);
        }
    }

    if (not has_driver(target_driver)) {
        throw exception{"xpu: Requested " + std::string{driver_str(target_driver)} + " device, but failed to load that driver."};
    }
    m_active_driver = target_driver;
    device_prop props;
    DRIVER_CALL(get_properties(&props, target_device));
    DRIVER_CALL(set_device(target_device));
    DRIVER_CALL(device_synchronize());

    if (m_active_driver != cpu) {
        XPU_LOG("Selected %s(arch = %d%d) as active device.", props.name.c_str(), props.major, props.minor);
    } else {
        XPU_LOG("Selected %s as active device.", props.name.c_str());
    }
}

void *runtime::host_malloc(size_t bytes) {
    void *ptr = nullptr;
    CPU_DRIVER_CALL(device_malloc(&ptr, bytes));
    return ptr;
}

void *runtime::device_malloc(size_t bytes) {
    if (logger::instance().active()) {
        size_t free, total;
        DRIVER_CALL(meminfo(&free, &total));
        device_prop props = device_properties();
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
    if (logger::instance().active()) {
        device_prop from = pointer_get_device(src);
        device_prop to = pointer_get_device(dst);
        XPU_LOG("Copy %lu bytes from %s to %s.", bytes, from.name.c_str(), to.name.c_str());
    }
    DRIVER_CALL(memcpy(dst, src, bytes));
}

void runtime::memset(void *dst, int ch, size_t bytes) {
    if (logger::instance().active()) {
        device_prop dev = pointer_get_device(dst);
        XPU_LOG("Setting %lu bytes on %s to %d.", bytes, dev.name.c_str(), ch);
    }
    DRIVER_CALL(memset(dst, ch, bytes));
}

xpu::device_prop runtime::device_properties() {
    int device;
    DRIVER_CALL(get_device(&device));
    device_prop props;
    DRIVER_CALL(get_properties(&props, device));
    return props;
}

bool runtime::has_driver(driver_t d) const {
    return get_driver(d) != nullptr;
}

xpu::device_prop runtime::pointer_get_device(const void *ptr) {

    for (driver_t driver_type : {cuda, hip, cpu}) {
        auto *driver = get_driver(driver_type);
        if (driver == nullptr) {
            continue;
        }
        int platform_device = 0;
        throw_on_driver_error(driver_type, driver->pointer_get_device(ptr, &platform_device));

        if (platform_device == -1) {
            continue;
        }

        return m_devices_by_driver[driver_type].at(platform_device);
    }

    // UNREACHABLE
    // cpu driver is always available
    throw exception{"Internal xpu error. This should never happen. Please file a bug."};
}

driver_interface *runtime::get_driver(driver_t d) const {
    switch (d) {
        case cpu: return m_cpu_driver.get();
        case cuda: return m_cuda_driver->obj;
        case hip: return m_hip_driver->obj;
    }

    return nullptr; // UNREACHABLE;
}

driver_interface *runtime::get_active_driver() const {
    return get_driver(m_active_driver);
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


