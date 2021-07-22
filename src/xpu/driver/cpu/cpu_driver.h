#ifndef XPU_DRIVER_CPU_CPU_DRIVER_H
#define XPU_DRIVER_CPU_CPU_DRIVER_H

#include "../../detail/driver_interface.h"

namespace xpu {
namespace detail {

class cpu_driver : public driver_interface {

public:
    virtual ~cpu_driver() {}

    detail::error setup(int) override;
    detail::error device_malloc(void **, size_t) override;
    detail::error free(void *) override;
    detail::error memcpy(void *, const void *, size_t) override;
    detail::error memset(void *, int, size_t) override;
    const char *error_to_string(error) override;

private:
    enum error_code : int {
        SUCCESS = 0,
        OUT_OF_MEMORY,
        INVALID_DEVICE,
    };

};

} // namespace detail
} // namespace xpu

#endif
