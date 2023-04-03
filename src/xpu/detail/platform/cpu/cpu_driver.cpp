#include "cpu_driver.h"

#include "../../log.h"

#include <unistd.h>

#include <cassert>
#include <cstdlib>
#include <cstring>

using namespace xpu::detail;

error cpu_driver::setup() {
    return SUCCESS;
}

error cpu_driver::malloc_device(void **ptr, size_t bytes) {
    *ptr = std::malloc(bytes);
    if (*ptr == nullptr) {
        return OUT_OF_MEMORY;
    }
    return SUCCESS;
}

error cpu_driver::malloc_host(void **ptr, size_t bytes) {
    *ptr = std::malloc(bytes);
    if (*ptr == nullptr) {
        return OUT_OF_MEMORY;
    }
    return SUCCESS;
}

error cpu_driver::malloc_shared(void **ptr, size_t bytes) {
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

error cpu_driver::create_queue(void **queue, int device) {
    *queue = nullptr;
    return (device == 0 ? SUCCESS : INVALID_DEVICE);
}

error cpu_driver::destroy_queue(void * /*queue*/) {
    return SUCCESS;
}

error cpu_driver::synchronize_queue(void * /*queue*/) {
    return SUCCESS;
}

error cpu_driver::memcpy(void *dst, const void *src, size_t bytes) {
    XPU_LOG("memcpy %lu bytes", bytes);
    std::memcpy(dst, src, bytes);
    return SUCCESS;
}

error cpu_driver::memcpy_async(void *dst, const void *src, size_t bytes, void * /*queue*/) {
    XPU_LOG("memcpy_async %lu bytes", bytes);
    std::memcpy(dst, src, bytes);
    return SUCCESS;
}

error cpu_driver::memset(void *dst, int ch, size_t bytes) {
    std::memset(dst, ch, bytes);
    return SUCCESS;
}

error cpu_driver::memset_async(void *dst, int ch, size_t bytes, void * /*queue*/) {
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
    props->arch = "";

    size_t free_mem, total_mem;
    meminfo(&free_mem, &total_mem);
    props->shared_mem_size = total_mem;
    props->const_mem_size = total_mem;

    props->warp_size = 1;
    props->max_threads_per_block = 1024;
    props->max_grid_size = {1024, 1024, 1024};

    return SUCCESS;
}

error cpu_driver::get_ptr_prop(const void * /*ptr*/, int *device, mem_type *type) {
    // There's no way to query the actual device here for the pointer
    // Therefore we have to assume it's pointing to cpu memory...
    *device = 0;
    *type = mem_unknown;
    return SUCCESS;
}

#ifdef __linux__
error cpu_driver::meminfo(size_t *free, size_t *total) {
    size_t pagesize = sysconf(_SC_PAGESIZE);
    *free = pagesize * sysconf(_SC_AVPHYS_PAGES);
    *total = pagesize * sysconf(_SC_PHYS_PAGES);
    return SUCCESS;
}
#elif defined __APPLE__
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <sys/types.h>
#include <sys/sysctl.h>
error cpu_driver::meminfo(size_t *free, size_t *total) {
    size_t pagesize = getpagesize();
    struct vm_statistics64 stats;
    mach_port_t host    = mach_host_self();
    natural_t   count   = HOST_VM_INFO64_COUNT;
    kern_return_t ret;
    if ((ret = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&stats, &count)) != KERN_SUCCESS) {
        return MACOSX_ERROR;
    }
    *free  = pagesize * stats.free_count;
    *total = pagesize * (stats.free_count     +
                         stats.active_count   +
                         stats.inactive_count +
                         stats.wire_count     +
                         stats.compressor_page_count);
    return SUCCESS;
}
#endif

const char *cpu_driver::error_to_string(error err) {
    switch (err) {
    case SUCCESS: return "Success";
    case OUT_OF_MEMORY: return "Out of memory";
    case INVALID_DEVICE: return "Invalid device";
    case MACOSX_ERROR: return "Macosx error";
    }

    return "Unknown error code";
}

xpu::detail::driver_t cpu_driver::get_type() {
    return cpu;
}
