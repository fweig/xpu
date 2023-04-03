#ifndef XPU_DRIVER_CPU_THIS_THREAD_H
#define XPU_DRIVER_CPU_THIS_THREAD_H

#include "../../../common.h"

namespace xpu::detail::this_thread {

extern thread_local dim block_idx;
extern thread_local dim grid_dim;

} // namespace xpu::detail::this_thread

#endif
