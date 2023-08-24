#ifndef XPU_DRIVER_SYCL_CMEM_IMPL_H
#define XPU_DRIVER_SYCL_CMEM_IMPL_H

#include "../../constant_memory.h"

#include <sycl/sycl.hpp>

#include <tuple>

namespace xpu {
template<typename...> class cmem;
} // namespace xpu

namespace xpu::detail {

template<typename C>
struct cmem_buffer {
    using buffer_t = sycl::buffer<typename C::data_t, 1>;
    buffer_t m_buffer;

    XPU_D cmem_buffer(const typename C::data_t *data)
        : m_buffer(data, sycl::range<1>{1}) {}
};

template<typename C>
struct cmem_accessor {
    using accessor_t = sycl::accessor<typename C::data_t, 0, sycl::access::mode::read, sycl::access::target::device>;
    accessor_t m_accessor;

    XPU_D cmem_accessor(cmem_buffer<C> &buffer, sycl::handler &handler)
        : m_accessor(buffer.m_buffer, handler) {}
};


template<typename>
struct cmem_traits{};

template<typename... Constants>
struct cmem_traits<cmem<Constants...>> {
    using buffer_tuple_t = std::tuple<cmem_buffer<Constants>...>;
    using accessor_tuple_t = std::tuple<cmem_accessor<Constants>...>;

    template<typename C>
    static cmem_accessor<C> make_cmem_accessor(cmem_buffer<C> &buffer, sycl::handler &handler) {
        return cmem_accessor<C>{buffer, handler};
    }

    static buffer_tuple_t make_buffers() {
        return std::make_tuple(cmem_buffer<Constants>{&constant_memory<Constants>}...);
    }

    static accessor_tuple_t make_accessors(buffer_tuple_t buffers, sycl::handler &handler) {
        return std::make_tuple(make_cmem_accessor<Constants>(std::get<cmem_buffer<Constants>>(buffers), handler)...);
    }
};

template<typename C>
class cmem_impl_leaf {

protected:
    using data_t = typename C::data_t;
    using accessor_t = typename cmem_accessor<C>::accessor_t;

    XPU_D cmem_impl_leaf(accessor_t accessor) : m_accessor(accessor) {}

    XPU_D const data_t &access() const { return m_accessor; }

private:
    accessor_t m_accessor;
};

template<typename Traits, typename...>
class cmem_impl_base {
private:
    using accessor_tuple_t = typename Traits::accessor_tuple_t;

protected:
    XPU_D cmem_impl_base(accessor_tuple_t &) {}
};

template<typename Traits, typename C, typename... ConstantsTail>
class cmem_impl_base<Traits, C, ConstantsTail...> : public cmem_impl_leaf<C>, public cmem_impl_base<Traits, ConstantsTail...> {

private:
    using accessor_tuple_t = typename Traits::accessor_tuple_t;

protected:
    XPU_D cmem_impl_base(accessor_tuple_t &accessors)
        : cmem_impl_leaf<C>(std::get<cmem_accessor<C>>(accessors).m_accessor)
        , cmem_impl_base<Traits, ConstantsTail...>(accessors) {}
};

template<typename... Constants>
class cmem_impl : public cmem_impl_base<cmem_traits<cmem<Constants...>>, Constants...> {

private:
    using traits = cmem_traits<cmem<Constants...>>;
    using accessor_tuple_t = typename traits::accessor_tuple_t;

public:
    XPU_D cmem_impl(accessor_tuple_t &accessors)
        : cmem_impl_base<traits, Constants...>(accessors) {}

    template<typename Constant>
    XPU_D std::enable_if_t<(std::is_same_v<Constant, Constants> || ...), const typename Constant::data_t &> get() const {
        return cmem_impl_leaf<Constant>::access();
    }
};

} // namespace xpu::detail

#endif
