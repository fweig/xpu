#include "cpu_driver.h"

#include <cstdlib>
#include <cstring>

using namespace xpu::detail;

error cpu_driver::setup() {
    return 0;
}

error cpu_driver::device_malloc(void ** ptr, size_t bytes) {
    *ptr = std::malloc(bytes);
    return *ptr == nullptr;
}

error cpu_driver::free(void *ptr) {
    std::free(ptr);
    return 0;
}

error cpu_driver::memcpy(void *dst, const void *src, size_t bytes) {
    std::memcpy(dst, src, bytes);
    return 0;
}

error cpu_driver::memset(void *dst, int ch, size_t bytes) {
    std::memset(dst, ch, bytes);
    return 0;
}
