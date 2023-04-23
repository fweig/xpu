#ifndef XPU_DETAIL_TIMERS_H
#define XPU_DETAIL_TIMERS_H

#include "common.h"

#include <string_view>

namespace xpu::detail {

void push_timer(std::string_view name);
timings pop_timer();

void add_memset_time(double);
void add_memcpy_time(double, direction_t);
void add_kernel_time(const char *, double);

} // namespace xpu::detail

#endif // XPU_DETAIL_TIMERS_H
