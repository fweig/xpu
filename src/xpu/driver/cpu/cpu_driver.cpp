#include "cpu_driver.h"

#include <cstdlib>
#include <cstring>

using namespace xpu;

error cpu_driver::setup() {
    return 0;
}

error cpu_driver::device_malloc(void ** ptr, size_t bytes) {
    *ptr = std::malloc(bytes); 
    return 0;
}

error cpu_driver::free(void *ptr) {
    std::free(ptr);
}

error cpu_driver::memcpy(void *dst, const void *src, size_t bytes) {
    std::memcpy(dst, src, bytes);
    return 0;
}