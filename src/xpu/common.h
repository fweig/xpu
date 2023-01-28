#ifndef XPU_COMMON_H
#define XPU_COMMON_H

#include "defines.h"

#include <string>

namespace xpu {

enum driver_t {
    cpu,
    cuda,
    hip,
};

struct device_prop {
    std::string name;
    driver_t driver;
    int major;
    int minor;
};

struct dim {
    int x = -1;
    int y = -1;
    int z = -1;

    dim() = default;
    constexpr dim(int _x) : x(_x) {}
    constexpr dim(int _x, int _y) : x(_x), y(_y) {}
    constexpr dim(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}

    constexpr int ndims() const { return 3 - (y < 0) - (z < 0); }

    constexpr int linear() const {
        return x * (y <= 0 ? 1 : y) * (z <= 0 ? 1 : z);
    }

    #if XPU_IS_HIP || XPU_IS_CUDA
    ::dim3 as_cuda_grid() const { return ::dim3{(unsigned int)x, (unsigned int)y, (unsigned int)z}; }
    #endif
};

struct grid {

    static inline grid n_blocks(dim nblocks);
    static inline grid n_threads(dim nthreads);

    dim nblocks;
    dim nthreads;

    inline void get_compute_grid(dim &block_dim, dim &grid_dim) const;

private:
    grid(dim b, dim t);

};

// FIXME: This really belongs into device.h, but can't be put there because of circular dependencies
template<typename SharedMemory>
class kernel_context {

public:
    using shared_memory = SharedMemory;
    // using constants = Constants;

    XPU_D kernel_context(shared_memory &smem) : m_smem(smem) {}

    XPU_D       shared_memory &smem()       { return m_smem; }
    XPU_D const shared_memory &smem() const { return m_smem; }

private:
    shared_memory &m_smem;
    // constants &m_cmem; TODO: implement later in preparation for SYCL backend
    // position &m_pos; TODO: implement later in preparation for SYCL backend

};

enum class side {
    host,
    device,
};

template<typename T, side S>
struct cmem_io {};

template<typename T, side S>
using cmem_io_t = typename cmem_io<T, S>::type;

template<typename T, side S>
struct cmem_device {};

template<typename T, side S>
using cmem_device_t = typename cmem_device<T, S>::type;

} // namespace xpu

#include "common_impl.h"

#endif
