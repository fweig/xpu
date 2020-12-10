#ifndef XPU_DRIVER_CPU_CPU_DRIVER_H
#define XPU_DRIVER_CPU_CPU_DRIVER_H

#include "../../host.h"

namespace xpu {

class cpu_driver : public driver_interface {

public:
    error setup() override;
    error device_malloc(void **, size_t) override;
    error free(void *) override;
    error memcpy(void *, const void *, size_t) override;
};

}

#endif