#include "backend.h"
#include "dl_utils.h"
#include "log.h"
#include "platform/cpu/cpu_driver.h"
#include <memory>
#include <stdexcept>

using namespace xpu::detail;

static std::unique_ptr<cpu_driver> the_cpu_driver;
static std::unique_ptr<lib_obj<backend_base>> the_cuda_driver;
static std::unique_ptr<lib_obj<backend_base>> the_hip_driver;
static std::unique_ptr<lib_obj<backend_base>> the_sycl_driver;

void backend::load(std::vector<driver_t> exclude) {
    auto should_exclude = [&exclude] (driver_t d) {
        return std::find(exclude.begin(), exclude.end(), d) != exclude.end();
    };

    if (should_exclude(cpu)) {
        throw std::runtime_error("backend::load: Cannot exclude cpu driver.");
    }

    XPU_LOG("Loading cpu driver.");
    the_cpu_driver = std::make_unique<cpu_driver>();
    call(cpu, &backend_base::setup);
    XPU_LOG("Finished loading cpu driver.");

    if (should_exclude(cuda)) {
        XPU_LOG("Cuda driver not active: Requested to skip driver.");
    } else {
        XPU_LOG("Loading cuda driver.");
        the_cuda_driver = std::make_unique<lib_obj<backend_base>>("libxpu_Cuda.so");
        if (the_cuda_driver->ok()) {
            call(cuda, &backend_base::setup);
            XPU_LOG("Finished loading cuda driver.");
        } else {
            XPU_LOG("Cuda driver not active: Couldn't find 'libxpu_Cuda.so'.");
        }
    }

    if (should_exclude(hip)) {
        XPU_LOG("Hip driver not active: Requested to skip driver.");
    } else {
        XPU_LOG("Loading hip driver.");
        the_hip_driver = std::make_unique<lib_obj<backend_base>>("libxpu_Hip.so");
        if (the_hip_driver->ok()) {
            call(hip, &backend_base::setup);
            XPU_LOG("Finished loading hip driver.");
        } else {
            XPU_LOG("Couldn't find 'libxpu_Hip.so'. Hip driver not active.");
        }
    }

    if (should_exclude(sycl)) {
        XPU_LOG("Sycl driver not active: Requested to skip driver.");
    } else {
        XPU_LOG("Loading sycl driver.");
        the_sycl_driver = std::make_unique<lib_obj<backend_base>>("libxpu_Sycl.so");
        if (the_sycl_driver->ok()) {
            call(sycl, &backend_base::setup);
            XPU_LOG("Finished loading sycl driver.");
        } else {
            XPU_LOG("Couldn't find 'libxpu_Sycl.so'. Sycl driver not active.");
        }
    }
}

void backend::unload(driver_t driver) {
    switch (driver) {
    case cpu:
        throw std::runtime_error("Cannot unload cpu driver.");
    case cuda:
        the_cuda_driver.reset();
        break;
    case hip:
        the_hip_driver.reset();
        break;
    case sycl:
        the_sycl_driver.reset();
        break;
    }
}

bool backend::is_available(driver_t driver) {
    return get(driver, false) != nullptr;
}

backend_base *backend::get(driver_t driver, bool throw_if_not_loaded /*= true*/) {
    backend_base *backend = nullptr;
    switch (driver) {
    case cpu:
        backend = the_cpu_driver.get();
        break;
    case cuda:
        if (the_cuda_driver != nullptr) {
            backend =  the_cuda_driver->obj;
        }
        break;
    case hip:
        if (the_hip_driver != nullptr) {
            backend = the_hip_driver->obj;
        }
        break;
    case sycl:
        if (the_sycl_driver != nullptr) {
            backend = the_sycl_driver->obj;
        }
        break;
    }
    if (backend == nullptr && throw_if_not_loaded) {
        XPU_LOG("Driver not loaded.");
        throw std::runtime_error("Driver not loaded.");
    }
    return backend;
}

void backend::raise_error(driver_t d, error err) {
    throw std::runtime_error(format("xpu: Driver '%s' raised error: %s (code %d)", driver_to_str(d), get(d)->error_to_string(err), err));
}
