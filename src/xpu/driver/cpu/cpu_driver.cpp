#include "cpu_driver.h"

#include "../../detail/log.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

using namespace xpu::detail;

error cpu_driver::setup(int device) {
    if (device != 0) {
        return INVALID_DEVICE;
    }
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

const char *cpu_driver::error_to_string(error err) {
    switch (err) {
    case SUCCESS: return "Success";
    case OUT_OF_MEMORY: return "Out of memory";
    case INVALID_DEVICE: return "Invalid device";
    }

    return "Unknown error code";
}
