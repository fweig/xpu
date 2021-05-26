#ifndef XPU_DRIVER_CPU_CPU_DRIVER_H
#define XPU_DRIVER_CPU_CPU_DRIVER_H

#include "../../detail/driver_interface.h"

namespace xpu {
namespace detail {

class cpu_driver : public driver_interface {

public:
    virtual ~cpu_driver() {}

    detail::error setup() override;
    detail::error device_malloc(void **, size_t) override;
    detail::error free(void *) override;
    detail::error memcpy(void *, const void *, size_t) override;
    detail::error memset(void *, int, size_t) override;
};

} // namespace detail
} // namespace xpu

#endif
