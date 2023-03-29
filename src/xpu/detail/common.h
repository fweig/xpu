#ifndef XPU_DETAIL_COMMON_H
#define XPU_DETAIL_COMMON_H

#include "../common.h"

#include <string>

namespace xpu::detail {

template<typename I, typename T>
struct action {
    using image = I;
    using tag = T;
};

enum mem_type {
    mem_host,
    mem_device,
    mem_shared,
    mem_unknown,
};

struct device {
    int id;
    driver_t backend;
    int device_nr;
};

struct device_prop {
    std::string name;
    std::string xpuid;
    driver_t driver;
    std::string arch;
};

using error = int;

} // namespace xpu::detail

#endif
