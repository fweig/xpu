#include "common.h"

const char *xpu::detail::driver_to_str(driver_t d, bool lower) {
    switch (d) {
    case cpu: return (lower ? "cpu" : "CPU");
    case cuda: return (lower ? "cuda" : "CUDA");
    case hip: return (lower ? "hip" : "HIP");
    case sycl: return (lower ? "sycl" : "SYCL");
    }
    return "unknown";
}
