#pragma once

#include <cstddef>
#include <utility>

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

namespace internal {

struct KernelTag {
    constexpr KernelTag(int, int) {}
};
constexpr KernelTag _kerneltag{0, 0};
}

void initialize(GPUBackendType);
void *malloc(size_t);
void free(void *);
void memcpy(void *, const void *, size_t);

template<typename T>
T *alloc(size_t N) {
    return static_cast<T *>(malloc(sizeof(T) * N));
}

template<typename KernelCls, typename... Args1, typename... Args2>
void runKernel(GPUError (KernelCls::*ptr)(internal::KernelTag, GPUKernelParams, Args2...), GPUKernelParams params, Args1&&... args) {
    (KernelCls::instance().*ptr)(internal::_kerneltag, params, std::forward<Args1>(args)...);
}

}