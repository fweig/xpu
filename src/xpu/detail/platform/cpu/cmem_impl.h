#ifndef XPU_DRIVER_CPU_CMEM_IMPL_H
#define XPU_DRIVER_CPU_CMEM_IMPL_H

#include "../../defines.h"
#include "../../constant_memory.h"

namespace xpu::detail {

template<typename C>
class cmem_impl_leaf {
protected:
    XPU_D const typename C::data_t &access() const { return constant_memory<C>; }
};

template<typename...>
class cmem_impl_base {};

template<typename C, typename... ConstantsTail>
class cmem_impl_base<C, ConstantsTail...> : public cmem_impl_leaf<C>, public cmem_impl_base<ConstantsTail...> {
};

template<typename... Constants>
class cmem_impl : public cmem_impl_base<Constants...> {
public:
    template<typename Constant>
    XPU_D std::enable_if_t<(std::is_same_v<Constant, Constants> || ...), const typename Constant::data_t &> get() const {
        return cmem_impl_leaf<Constant>::access();
    }
};

} // namespace xpu::detail

#endif
