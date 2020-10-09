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
const char *getName();

template<typename Kernel, typename Enable = typename std::enable_if<is_kernel<Kernel>::value>::type, typename... Args>
void runKernel(GPUKernelParams params, Args&&... args) {
    std::string backend = "CPU";
    if (activeBackend() == GPUBackendType::CUDA) {
        backend = "CUDA";
    }
    std::cout << "Running kernel " << getName<Kernel>() << " on backend " << backend << std::endl;
    Kernel::dispatch(Kernel::Library::instance(activeBackend()), params, std::forward<Args>(args)...);
}

}