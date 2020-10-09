#pragma once

#include "../../macros.h"

#define KERNEL_IMPL(name, sharedMemoryT, ...) \
    void kernel_ ## name(const GPUKernelInfo &, sharedMemoryT &, PARAM_LIST(__VA_ARGS__)); \
    GPUError TestKernelsCPU::run_ ## name(GPUKernelParams params, PARAM_LIST(__VA_ARGS__)) { \
        for (int i = 0; i < params.range.x; i++) { \
            sharedMemoryT shm{}; \
            GPUKernelInfo info{ \
            .threadIdx = {0, 0, 0}, \
            .nThreads  = {1, 0, 0}, \
            .blockIdx  = {i, 0, 0}, \
            .nBlocks   = {params.range.x, 0, 0}}; \
            \
            kernel_ ## name(info, shm, PARAM_NAMES(__VA_ARGS__)); \
        } \
        return 0; \
    } \
    \
    void kernel_ ## name(const GPUKernelInfo &info, sharedMemoryT &shm, PARAM_LIST(__VA_ARGS__))
