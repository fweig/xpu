#include "common.h"

#include <algorithm>

const char *xpu::detail::driver_to_str(driver_t d, bool lower) {
    switch (d) {
    case cpu: return (lower ? "cpu" : "CPU");
    case cuda: return (lower ? "cuda" : "CUDA");
    case hip: return (lower ? "hip" : "HIP");
    case sycl: return (lower ? "sycl" : "SYCL");
    }
    return "unknown";
}

void xpu::detail::timings::merge(const timings &other) {
    wall += other.wall;
    has_details |= other.has_details;
    copy_h2d += other.copy_h2d;
    bytes_h2d += other.bytes_h2d;
    copy_d2h += other.copy_d2h;
    bytes_d2h += other.bytes_d2h;
    memset += other.memset;
    bytes_memset += other.bytes_memset;
    bytes_input += other.bytes_input;

    for (auto &k : other.kernels) {
        auto it = std::find_if(kernels.begin(), kernels.end(), [&](const auto &k2) {
            return k2.name == k.name;
        });
        if (it == kernels.end()) {
            kernels.push_back(k);
        } else {
            it->times.insert(it->times.end(), k.times.begin(), k.times.end());
            it->bytes_input += k.bytes_input;
        }
    }

    for (auto &c : other.children) {
        auto it = std::find_if(children.begin(), children.end(), [&](const auto &c2) {
            return c2.name == c.name;
        });
        if (it == children.end()) {
            children.push_back(c);
        } else {
            it->merge(c);
        }
    }
}
