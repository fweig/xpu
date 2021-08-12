#include "cpu_driver.h"

#include "../../detail/log.h"
#include "../../host.h"

#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <cstring>

using namespace xpu::detail;

error cpu_driver::setup() {
    return SUCCESS;
}

error cpu_driver::device_malloc(void ** ptr, size_t bytes) {
    *ptr = std::malloc(bytes);
    if (*ptr == nullptr) {
        return OUT_OF_MEMORY;
    }
    return SUCCESS;
}

error cpu_driver::free(void *ptr) {
    std::free(ptr);
    return SUCCESS;
}

error cpu_driver::memcpy(void *dst, const void *src, size_t bytes) {
    XPU_LOG("memcpy %lu bytes", bytes);
    std::memcpy(dst, src, bytes);
    return SUCCESS;
}

error cpu_driver::memset(void *dst, int ch, size_t bytes) {
    std::memset(dst, ch, bytes);
    return SUCCESS;
}

error cpu_driver::num_devices(int *devices) {
    *devices = 1;
    return SUCCESS;
}

error cpu_driver::set_device(int device) {
    return (device == 0 ? SUCCESS : INVALID_DEVICE);
}

error cpu_driver::get_device(int *device) {
    *device = 0;
    return SUCCESS;
}

error cpu_driver::device_synchronize() {
    return SUCCESS;
}

error cpu_driver::get_properties(device_prop *props, int device) {
    if (device != 0) {
        return INVALID_DEVICE;
    }

    props->name = "CPU";
    props->driver = cpu;
    props->major = 0;
    props->minor = 0;

    return SUCCESS;
}

error cpu_driver::meminfo(size_t *free, size_t *total) {
    size_t pagesize = sysconf(_SC_PAGESIZE);
    *free = pagesize * sysconf(_SC_AVPHYS_PAGES);
    *total = pagesize * sysconf(_SC_PHYS_PAGES);
    return SUCCESS;
}

const char *cpu_driver::error_to_string(error err) {
    switch (err) {
    case SUCCESS: return "Success";
    case OUT_OF_MEMORY: return "Out of memory";
    case INVALID_DEVICE: return "Invalid device";
    }

    return "Unknown error code";
}

xpu::driver_t cpu_driver::get_type() {
    return xpu::cpu;
}
