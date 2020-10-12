#pragma once

#include <cstddef>
#include <iostream>
#include <utility>
#include <type_traits>

enum class GPUBackendType {
    CPU,
    CUDA,
};

struct Vec3i { int x; int y; int z; };

// TODO rename + add lane info
// Common calls: 
// - num of blocks + (optional) lane
// - num of threads + (optional) lane
// - auto num of blocks + (optional) lane
// xpu::grid::n_blocks(xpu::dim, xpu::lane lane=xpu::lane::default)
// xpu::grid::n_threads(xpu::dim, xpu::lane lane=xpu::lane::default)
// xpu::grid::auto(xpu::lane lane=xpu::lane::default)
// xpu::run_kernel<VectorOps::add>(xpu::grid::n_threads(100), )

struct GPUKernelParams {
    Vec3i range;
};

struct GPUKernelInfo {
    Vec3i threadIdx;
    Vec3i nThreads;
    Vec3i blockIdx;
    Vec3i nBlocks;
};

using GPUError = int;

namespace gpu {

template<class K>
struct is_kernel : std::false_type {};

template<class L>
struct Kernel {
    using Library = L;
};

void initialize(GPUBackendType);
void *malloc(size_t);
void free(void *);
void memcpy(void *, const void *, size_t);

GPUBackendType activeBackend();

template<typename T>
T *alloc(size_t N) {
    return static_cast<T *>(malloc(sizeof(T) * N));
}

template<typename Kernel, typename Enable = typename std::enable_if<is_kernel<Kernel>::value>::type>
const char *get_name() {
    return Kernel::name();
}

template<typename Kernel, typename Enable = typename std::enable_if<is_kernel<Kernel>::value>::type, typename... Args>
void runKernel(GPUKernelParams params, Args&&... args) {
    std::string backend = "CPU";
    if (activeBackend() == GPUBackendType::CUDA) {
        backend = "CUDA";
    }
    std::cout << "Running kernel " << get_name<Kernel>() << " on backend " << backend << std::endl;
    Kernel::dispatch(Kernel::Library::instance(activeBackend()), params, std::forward<Args>(args)...);
}

}