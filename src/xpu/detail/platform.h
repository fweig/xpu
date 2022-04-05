#ifndef XPU_DETAIL_PLATFORM_H
#define XPU_DETAIL_PLATFORM_H

#include <cstddef>

namespace xpu {
namespace detail {

void *dlopen_image(const char *name);
void get_meminfo(size_t *free, size_t *total);

} // namespace detail
} // namespace xpu

#endif
