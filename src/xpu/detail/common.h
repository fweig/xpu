#ifndef XPU_DETAIL_COMMON_H
#define XPU_DETAIL_COMMON_H

#include "../common.h"

#include <array>
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
    // Filled by driver
    std::string name;
    driver_t driver;
    std::string arch;
    size_t shared_mem_size;
    size_t const_mem_size;

    size_t warp_size;
    size_t max_threads_per_block;
    std::array<size_t, 3> max_grid_size;

    // Filled by runtime
    std::string xpuid;
    int id;
    int device_nr;

    size_t global_mem_total;
    size_t global_mem_available;
};

using error = int;

} // namespace xpu::detail

#endif
