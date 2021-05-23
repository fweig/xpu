#ifndef XPU_DRIVER_CPU_THIS_THREAD_H
#define XPU_DRIVER_CPU_THIS_THREAD_H

#include "../../common.h"

namespace xpu {
namespace detail {
namespace this_thread {
extern thread_local dim block_idx;
extern thread_local dim grid_dim;
} // namespace this_thread
} // namespace detail
} // namespace xpu

#endif
