#ifndef XPU_DETAIL_CONSTANT_MEMORY_H
#define XPU_DETAIL_CONSTANT_MEMORY_H

#include "../defines.h"

namespace xpu::detail {

template<typename A>
XPU_DETAIL_CONSTANT_SPEC typename A::data_t constant_memory;

} // namespace xpu::detail

#endif
