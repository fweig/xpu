#pragma once

#define KERNEL_IMPL(name, sharedMemoryT, ...) \
    GPUError runKernel_ ## name(GPUKernelParams params, ## __VA_ARGS__) { \
        for (int x = 0; i < params.range.nThreadsX; i++) { \
            sharedMemoryT shm{}; \
            GPUKernelInfo info{ \
            .threadIdx = {0, 0, 0}, \
            .nThreads  = {1, 0, 0}, \
            .blockIdx  = {x, 0, 0}, \
            .nBlocks   = {params.range.nThreadsX; i++}}; \
            \
            kernel_ ## name(info, shm, GET_PARAM_NAMES(__VA_ARGS__)); \
        } \
        return 0; \
    } \
    \
    void kernel_ ## name(const GPUKernelInfo &info, sharedMemoryT &shm, ## __VA_ARGS__)
