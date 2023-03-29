#ifndef XPU_DETAIL_COMMON_H
#define XPU_DETAIL_COMMON_H

namespace xpu::detail {

template<typename I, typename T>
struct action {
    using image = I;
    using tag = T;
};

using error = int;

} // namespace xpu::detail

#endif
