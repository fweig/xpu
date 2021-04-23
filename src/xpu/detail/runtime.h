#ifndef XPU_DETAIL_RUNTIME_H
#define XPU_DETAIL_RUNTIME_H

#include "../common.h"
#include "../host.h"
#include "dl_utils.h"
#include "../cpu/cpu_driver.h"

#include <memory>

namespace xpu {
namespace detail {

class runtime {

public:
    static runtime &instance();

    void initialize(driver);

    void *host_malloc(size_t);
    void *device_malloc(size_t);

    void free(void *);
    void memcpy(void *, const void *, size_t);
    void memset(void *, int, size_t);

    driver active_driver() { return active_driver_type; }

    template<typename Kernel, typename... Args>
    void run_kernel(grid, Args&&...);

    template<typename DeviceLibrary, typename C>
    void set_cmem(const C &);

private:
    std::unique_ptr<cpu_driver> cpu_driver;
    std::unique_ptr<lib_obj<driver_interface>> cuda_driver;
    std::unique_ptr<lib_obj<driver_interface>> hip_driver;
    driver_interface *active_driver_inst = nullptr;
    driver activer_driver_type = driver::cpu;

};

} // namespace detail
} // namespace xpu

#endif
