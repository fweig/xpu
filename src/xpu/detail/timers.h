#ifndef XPU_DETAIL_TIMERS_H
#define XPU_DETAIL_TIMERS_H

#include "common.h"

#include <string_view>

namespace xpu::detail {

void push_timer(std::string_view name);
timings pop_timer();

void add_memset_time(double, size_t);
void add_memcpy_time(double, direction_t, size_t);
void add_kernel_time(std::string_view, double);

void add_bytes_timer(size_t);
void add_bytes_kernel(std::string_view, size_t);

} // namespace xpu::detail

#endif // XPU_DETAIL_TIMERS_H
