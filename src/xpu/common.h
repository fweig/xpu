#ifndef XPU_COMMON_H
#define XPU_COMMON_H

#include "detail/common.h"
#include "defines.h"
#include "detail/buffer_registry.h"

#include <string>

namespace xpu {

enum driver_t {
    cpu = detail::driver_t::cpu,
    cuda = detail::cuda,
    hip = detail::hip,
    sycl = detail::sycl,
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

enum buffer_type {
    host_buffer = detail::host_buffer,
    device_buffer = detail::device_buffer,
    shared_buffer = detail::shared_buffer,
    io_buffer   = detail::io_buffer,
};

template<typename T>
class buffer {

public:
    using value_type = T;

    /**
     * @brief Create an emtpy buffer.
     */
    buffer() = default;

    /**
     * @brief Create a buffer with the given size.
     * @param N Size of the buffer.
     * @param type Type of the buffer.
     * @param data Pointer to the data to use for the buffer.
     *
     * Allocates a buffer of the given size and type.
     * Behavior depends on the type of the buffer:
     * - host_buffer:
     *     Allocates a buffer in host memory. The buffer is accessible from the device.
     *     If data is not null, the buffer is initialized with the data.
     * - device_buffer:
     *     Allocates a buffer in device memory that is not accessible from the host.
     *     Memory is not initialized and 'data' pointer has no effect
     * - shared_buffer:
     *     Allocates a shared (managed) buffer that is accessible from the host and the device.
     *     If data is not null, the buffer is initialized with the data.
     * - io_buffer:
     *     Allocates a buffer in device memory. Excepts that data points to a memory region with at least N elements.
     *     Buffer may be copied to / from the device using xpu::copy from / to 'data'.
     *     Note: If the device is a CPU, the underlying pointer simply points to 'data' and no additional allocation takes place.
     *     xpu::copy calls become no-ops in this case.
     */
    buffer(size_t N, buffer_type type, T *data = nullptr);

    /**
     * @brief Free the buffer.
     */
    XPU_H XPU_D ~buffer();

    XPU_H XPU_D buffer(const buffer &other);
    XPU_H XPU_D buffer(buffer &&other);
    XPU_H XPU_D buffer &operator=(const buffer &other);
    XPU_H XPU_D buffer &operator=(buffer &&other);

    template<typename U>
    XPU_H XPU_D buffer(const buffer<U> &other);

    template<typename U>
    XPU_H XPU_D buffer(buffer<U> &&other);

    template<typename U>
    XPU_H XPU_D buffer &operator=(const buffer<U> &other);

    template<typename U>
    XPU_H XPU_D buffer &operator=(buffer<U> &&other);

    XPU_H XPU_D void reset();
    void reset(size_t N, buffer_type type, T *data = nullptr);

    XPU_H XPU_D T *get() const { return m_data; }

    XPU_H XPU_D T &operator*() const { return *m_data; }
    XPU_H XPU_D T *operator->() const { return m_data; }

    XPU_H XPU_D T &operator[](size_t i) const { return m_data[i]; }

private:
    T *m_data = nullptr;

    XPU_H XPU_D void add_ref();
    XPU_H XPU_D void remove_ref();
};

} // namespace xpu

#include "common_impl.h"

#endif
