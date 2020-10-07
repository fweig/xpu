#pragma once

#include <cstddef>

enum class GPUBackendType {
    GPU,
    CUDA,
};

namespace gpu {
void initialize(GPUBackendType);
void *malloc(size_t);
void free(void *);
void memcpy(void *, const void *, size_t);
}
