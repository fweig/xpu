#include "runtime.h"
#include "../host.h"

#include <cstdlib>
#include <sstream>

#define DRIVER_CALL_I(type, func) throw_on_driver_error((type), get_driver((type))->func)
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

    XPU_LOG("Loading cpu driver.");
    m_cpu_driver.reset(new cpu_driver{});
    CPU_DRIVER_CALL(setup());
    XPU_LOG("Finished loading cuda driver.");

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

    XPU_LOG("Loading sycl driver.");
    m_sycl_driver.reset(new lib_obj<driver_interface>{"libxpu_Sycl.so"});
    if (m_sycl_driver->ok()) {
        DRIVER_CALL_I(sycl, setup());
        XPU_LOG("Finished loading sycl driver.");
    } else {
        XPU_LOG("Couldn't find 'libxpu_Sycl.so'. Sycl driver not active.");
    }

    XPU_LOG("Found devices:");
    for (driver_t driver : {cpu, cuda, hip, sycl}) {
        if (not has_driver(driver)) {
            XPU_LOG("  No %s devices found.", driver_str(driver));
            continue;
        }
        int ndevices = 0;
        DRIVER_CALL_I(driver, num_devices(&ndevices));
        XPU_LOG(" %s (%d)", driver_str(driver), ndevices);
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
    XPU_LOG("Allocating %lu bytes @ address %p on host memory with driver %s.", bytes, ptr, driver_str(m_active_device.backend));
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

void *runtime::create_queue(device dev) {
    void *queue = nullptr;
    DRIVER_CALL_I(dev.backend, create_queue(&queue, dev.device_nr));
    return queue;
}

void runtime::destroy_queue(queue_handle queue) {
    DRIVER_CALL_I(queue.dev.backend, destroy_queue(queue.handle));
}

void runtime::synchronize_queue(queue_handle queue) {
    DRIVER_CALL_I(queue.dev.backend, synchronize_queue(queue.handle));
}

void runtime::memcpy(void *dst, const void *src, size_t bytes) {
    if (logger::instance().active()) {
        ptr_prop src_prop{src};
        ptr_prop dst_prop{dst};

        device_prop from;
        DRIVER_CALL_I(src_prop.backend(), get_properties(&from, src_prop.device()));

        device_prop to;
        DRIVER_CALL_I(dst_prop.backend(), get_properties(&to, dst_prop.device()));

        XPU_LOG("Copy %lu bytes from %s to %s.", bytes, from.name.c_str(), to.name.c_str());
    }
    DRIVER_CALL(memcpy(dst, src, bytes));
}

void runtime::memset(void *dst, int ch, size_t bytes) {
    if (logger::instance().active()) {
        ptr_prop dst_prop{dst};
        device_prop dev;
        DRIVER_CALL_I(dst_prop.backend(), get_properties(&dev, dst_prop.device()));
        XPU_LOG("Setting %lu bytes on %s to %d.", bytes, dev.name.c_str(), ch);
    }
    DRIVER_CALL(memset(dst, ch, bytes));
}

xpu::detail::device_prop runtime::device_properties(int id) {
    detail::device_prop props;
    detail::device d = m_devices.at(id);
    DRIVER_CALL_I(d.backend, get_properties(&props, d.device_nr));

    props.xpuid = detail::format("%s%d", driver_str(d.backend, true), d.device_nr);
    props.id = d.id;
    props.device_nr = d.device_nr;
    DRIVER_CALL_I(d.backend, meminfo(&props.global_mem_available, &props.global_mem_total));

    return props;
}

std::optional<std::pair<xpu::driver_t, int>> runtime::try_parse_device(std::string_view device_name) const {
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

bool runtime::has_driver(driver_t d) const {
    return get_driver(d) != nullptr;
}

device runtime::get_device(driver_t d, int device_nr) const {
    auto it = std::find_if(m_devices.begin(), m_devices.end(), [d, device_nr](const device &dev) {
        return dev.backend == d and dev.device_nr == device_nr;
    });

    if (it != m_devices.end()) {
        return *it;
    }

    raise_error(format("Requested device %s%d does not exist.", driver_str(d, true), device_nr));
}

device runtime::get_device(std::string_view device_name) const {
    auto dev = try_parse_device(device_name);
    if (dev == std::nullopt) {
        raise_error(format("Invalid device name '%*.s'.", static_cast<int>(device_name.size()), device_name.data()));
    }
    return get_device(dev->first, dev->second);
}

void runtime::get_ptr_prop(const void *ptr, ptr_prop *prop) {

    prop->m_ptr = const_cast<void *>(ptr);

    for (driver_t driver_type : {cuda, hip, sycl, cpu}) {
        auto *driver = get_driver(driver_type);
        if (driver == nullptr) {
            continue;
        }
        int platform_device = 0;
        detail::mem_type mem_type;
        throw_on_driver_error(driver_type, driver->get_ptr_prop(ptr, &platform_device, &mem_type));

        if (platform_device == -1) {
            continue;
        }

        prop->m_type = static_cast<xpu::mem_type>(mem_type);
        prop->m_backend = driver_type;
        prop->m_device = platform_device;
        return;
    }

    // UNREACHABLE
    // cpu driver will always return a valid device,
    // that's why it needs to be the last driver to be checked.
    RAISE_INTERNAL_ERROR();
}

driver_interface *runtime::get_driver(driver_t d) const {
    switch (d) {
        case cpu: return m_cpu_driver.get();
        case cuda: return m_cuda_driver->obj;
        case hip: return m_hip_driver->obj;
        case sycl: return m_sycl_driver->obj;
    }

    RAISE_INTERNAL_ERROR();
}

driver_interface *runtime::get_active_driver() const {
    return get_driver(m_active_device.backend);
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

const char *runtime::driver_str(driver_t d, bool lower) const {
    switch (d) {
    case cpu: return (lower ? "cpu" : "CPU");
    case cuda: return (lower ? "cuda" : "CUDA");
    case hip: return (lower ? "hip" : "HIP");
    case sycl: return (lower ? "sycl" : "SYCL");
    }
    RAISE_INTERNAL_ERROR();
}

void runtime::throw_on_driver_error(driver_t d, error err) const {
    if (err == 0) {
        return;
    }

    raise_error(format("xpu: Driver '%s' raised error: %s (code %d)", driver_str(d), get_driver(d)->error_to_string(err), err));
}

void runtime::raise_error_if(bool condition, std::string_view error_msg) const {
    if (condition) {
        raise_error(error_msg);
    }
}

[[noreturn]] void runtime::raise_error(std::string_view error_msg) const {
    throw xpu::exception(error_msg);
}
