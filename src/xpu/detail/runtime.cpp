#include "runtime.h"
#include "backend.h"
#include "../host.h"

#include <cstdlib>
#include <sstream>

#define DRIVER_CALL_I(type, func) throw_on_driver_error(static_cast<detail::driver_t>(type), \
    backend::get(static_cast<detail::driver_t>(type))->func)
#define DRIVER_CALL(func) DRIVER_CALL_I(m_active_device.backend, func)
#define CPU_DRIVER_CALL(func) DRIVER_CALL_I(cpu, func)
#define RAISE_INTERNAL_ERROR() raise_error(format("%s:%d: Internal xpu error. This should never happen! Please file a bug.", __FILE__, __LINE__))

using namespace xpu::detail;

bool runtime::getenv_bool(std::string driver_name, bool fallback) {
    const char *env = getenv(driver_name.c_str());
    return (env == nullptr ? fallback : (strcmp(env, "0") != 0));
}

std::string runtime::getenv_str(std::string driver_name, std::string_view fallback) {
    const char *env = getenv(driver_name.c_str());
    return (env == nullptr ? std::string{fallback} : std::string{env});
}

runtime &runtime::instance() {
    static runtime the_runtime{};
    return the_runtime;
}

void runtime::initialize(const settings &settings) {

    bool verbose = getenv_bool("XPU_VERBOSE", settings.verbose);
    if (verbose) {
        logger::instance().initialize(settings.logging_sink);
    }

    m_measure_time = getenv_bool("XPU_PROFILE", settings.profile);

    backend::load();

    XPU_LOG("Found devices:");
    for (driver_t driver : {cpu, cuda, hip, sycl}) {
        if (not backend::is_available(driver)) {
            XPU_LOG("  No %s devices found.", driver_to_str(driver));
            continue;
        }
        int ndevices = 0;
        DRIVER_CALL_I(driver, num_devices(&ndevices));
        XPU_LOG(" %s (%d)", driver_to_str(driver), ndevices);
        for (int i = 0; i < ndevices; i++) {
            device dev;
            dev.backend = driver;
            dev.device_nr = i;
            dev.id = m_devices.size();
            m_devices.emplace_back(dev);
        }
    }

    std::optional<detail::device> target_device;
    if (auto device_env = getenv_str("XPU_DEVICE", settings.device); true) {
        target_device = get_device(device_env);

        if (target_device == std::nullopt) {
            raise_error(format("Requested unknown driver with environment variable XPU_DEVICE='%s'", device_env.c_str()));
        }
    }

    m_active_device = *target_device;
    xpu::detail::device_prop props;
    DRIVER_CALL(get_properties(&props, m_active_device.device_nr));
    DRIVER_CALL(set_device(m_active_device.device_nr));
    DRIVER_CALL(device_synchronize());

    if (m_active_device.backend != cpu) {
        XPU_LOG("Selected %s(arch = %s) as active device. (id = %d)", props.name.c_str(), props.arch.c_str(), m_active_device.id);
    } else {
        XPU_LOG("Selected %s as active device.", props.name.c_str());
    }
}

void *runtime::malloc_host(size_t bytes) {
    void *ptr = nullptr;
    DRIVER_CALL(malloc_host(&ptr, bytes));
    XPU_LOG("Allocating %lu bytes @ address %p on host memory with driver %s.", bytes, ptr, driver_to_str(m_active_device.backend));
    return ptr;
}

void *runtime::malloc_device(size_t bytes) {
    if (logger::instance().active()) {
        size_t free, total;
        DRIVER_CALL(meminfo(&free, &total));
        device_prop props = device_properties(m_active_device.id);
        XPU_LOG("Allocating %lu bytes on device %s. [%lu / %lu available]", bytes, props.name.c_str(), free, total);
    }
    void *ptr = nullptr;
    DRIVER_CALL(malloc_device(&ptr, bytes));
    return ptr;
}

void *runtime::malloc_shared(size_t bytes) {
    if (logger::instance().active()) {
        size_t free, total;
        DRIVER_CALL(meminfo(&free, &total));
        device_prop props = device_properties(m_active_device.id);
        XPU_LOG("Allocating %lu bytes of managed memory on device %s. [%lu / %lu available]", bytes, props.name.c_str(), free, total);
    }
    void *ptr = nullptr;
    DRIVER_CALL(malloc_shared(&ptr, bytes));
    return ptr;
}

void runtime::free(void *ptr) {
    DRIVER_CALL(free(ptr));
}

void runtime::memcpy(void *dst, const void *src, size_t bytes) {
    if (logger::instance().active()) {
        xpu::ptr_prop src_prop{src};
        xpu::ptr_prop dst_prop{dst};

        device_prop from;
        DRIVER_CALL_I(src_prop.backend(), get_properties(&from, src_prop.device().device_nr()));

        device_prop to;
        DRIVER_CALL_I(dst_prop.backend(), get_properties(&to, dst_prop.device().device_nr()));

        XPU_LOG("Copy %lu bytes from %s to %s.", bytes, from.name.c_str(), to.name.c_str());
    }
    DRIVER_CALL(memcpy(dst, src, bytes));
}

void runtime::memset(void *dst, int ch, size_t bytes) {
    if (logger::instance().active()) {
        xpu::ptr_prop dst_prop{dst};
        device_prop dev;
        DRIVER_CALL_I(dst_prop.backend(), get_properties(&dev, dst_prop.device().device_nr()));
        XPU_LOG("Setting %lu bytes on %s to %d.", bytes, dev.name.c_str(), ch);
    }
    DRIVER_CALL(memset(dst, ch, bytes));
}

xpu::detail::device_prop runtime::device_properties(int id) {
    detail::device_prop props;
    detail::device d = m_devices.at(id);
    DRIVER_CALL_I(d.backend, get_properties(&props, d.device_nr));

    props.xpuid = detail::format("%s%d", driver_to_str(d.backend, true), d.device_nr);
    props.id = d.id;
    props.device_nr = d.device_nr;
    DRIVER_CALL_I(d.backend, meminfo(&props.global_mem_available, &props.global_mem_total));

    return props;
}

std::optional<std::pair<xpu::detail::driver_t, int>> runtime::try_parse_device(std::string_view device_name) const {
    std::vector<std::pair<std::string, driver_t>> str_to_driver {
        {"cpu", cpu},
        {"cuda", cuda},
        {"hip", hip},
        {"sycl", sycl},
    };

    bool valid_driver = false;
    int target_device = 0;
    driver_t target_driver = cpu;
    for (auto &driver_str : str_to_driver) {
        const std::string &driver_name = driver_str.first;
        if (strncmp(device_name.data(), driver_name.c_str(), driver_name.size()) == 0) {
            valid_driver = true;
            target_driver = driver_str.second;

            if (device_name == driver_name) {
                target_device = 0;
            } else {
                std::string device_id{device_name.substr(driver_name.size())};
                try {
                    target_device = std::stoi(device_id);
                } catch (std::exception &e) {
                    raise_error(format("Invalid device ID '%*.s'", static_cast<int>(device_name.size()), device_name.data()));
                }
            }

            break;
        }
    }

    if (not valid_driver) {
        raise_error(format("Requested unknown driver '%*.s'", static_cast<int>(device_name.size()), device_name.data()));
    }

    return std::make_pair(target_driver, target_device);
}

device runtime::get_device(driver_t d, int device_nr) const {
    auto it = std::find_if(m_devices.begin(), m_devices.end(), [d, device_nr](const device &dev) {
        return dev.backend == d and dev.device_nr == device_nr;
    });

    if (it != m_devices.end()) {
        return *it;
    }

    raise_error(format("Requested device %s%d does not exist.", driver_to_str(d, true), device_nr));
}

device runtime::get_device(std::string_view device_name) const {
    auto dev = try_parse_device(device_name);
    if (dev == std::nullopt) {
        raise_error(format("Invalid device name '%*.s'.", static_cast<int>(device_name.size()), device_name.data()));
    }
    return get_device(dev->first, dev->second);
}

void runtime::get_ptr_prop(const void *ptr, ptr_prop *prop) {

    prop->ptr = const_cast<void *>(ptr);

    for (driver_t driver_type : {cuda, hip, sycl, cpu}) {
        if (not backend::is_available(driver_type)) {
            continue;
        }
        int platform_device = 0;
        detail::mem_type mem_type;
        backend::call(driver_type, &backend_base::get_ptr_prop, ptr, &platform_device, &mem_type);

        if (platform_device == -1) {
            continue;
        }

        prop->type = mem_type;
        prop->dev = get_device(driver_type, platform_device);
        return;
    }

    // UNREACHABLE
    // cpu driver will always return a valid device,
    // that's why it needs to be the last driver to be checked.
    RAISE_INTERNAL_ERROR();
}

std::string runtime::complete_file_name(const char *fdriver_name, driver_t d) const {
    std::string prefix = "lib";
    std::string suffix = "";
    switch (d) {
    case cuda: suffix = "_Cuda.so"; break;
    case hip:  suffix = "_Hip.so"; break;
    case sycl: suffix = "_Sycl.so"; break;
    case cpu:  suffix = ".so"; break;
    }
    return prefix + std::string{fdriver_name} + suffix;
}

void runtime::throw_on_driver_error(driver_t d, error err) const {
    if (err == 0) {
        return;
    }

    raise_error(format("xpu: Driver '%s' raised error: %s (code %d)", driver_to_str(d), backend::get(d)->error_to_string(err), err));
}

void runtime::raise_error_if(bool condition, std::string_view error_msg) const {
    if (condition) {
        raise_error(error_msg);
    }
}

[[noreturn]] void runtime::raise_error(std::string_view error_msg) const {
    throw xpu::exception(error_msg);
}
