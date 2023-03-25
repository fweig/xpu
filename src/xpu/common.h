#ifndef XPU_COMMON_H
#define XPU_COMMON_H

#include "defines.h"

#include <string>

namespace xpu {

enum driver_t {
    cpu,
    cuda,
    hip,
    sycl,
};

constexpr inline size_t num_drivers = 4;

struct device_prop {
    std::string name;
    std::string xpuid;
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

/**
 * @brief 3d execution grid describing the number of blocks and threads of a kernel
 * Use 'n_blocks' or 'n_threads' to construct a grid.
 */
struct grid {

    dim nblocks;
    dim nthreads;

    inline void get_compute_grid(dim &block_dim, dim &grid_dim) const;

private:
    friend inline grid n_blocks(dim);
    friend inline grid n_threads(dim);
    grid(dim b, dim t);

};

/**
 * @brief Construct a grid with the given number of blocks in each dimension
 */
inline grid n_blocks(dim nblocks);

/**
 * @brief Construct a grid with the given number of threads in each dimension
 * If the number of threads is not a multiple of the block size, the grid size
 * will be rounded up to the next multiple of the block size.
 */
inline grid n_threads(dim nthreads);

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
