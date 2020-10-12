#pragma once

#include "../../macros.h"

#include <iostream>

// TODO: don't hardcode block size
#define KERNEL_IMPL(name, sharedMemoryT, ...) \
    __device__ void name ## _impl(const GPUKernelInfo &info, sharedMemoryT &, PARAM_LIST(__VA_ARGS__)); \
    __global__ void name ## _entry(PARAM_LIST(__VA_ARGS__)) { \
        __shared__ sharedMemoryT shm; \
        GPUKernelInfo info{ \
            .threadIdx = {int(threadIdx.x), 0, 0}, \
            .nThreads  = {int(blockDim.x), 0, 0}, \
            .blockIdx  = {int(blockIdx.x), 0, 0}, \
            .nBlocks   = {int(gridDim.x), 0, 0} \
        }; \
        name ## _impl(info, shm, PARAM_NAMES(__VA_ARGS__)); \
    } \
    GPUError XPU_DEVICE_LIBRARY_BACKEND_NAME::run_ ## name(GPUKernelParams params, PARAM_LIST(__VA_ARGS__)) { \
        name ## _entry<<<(params.range.x + 31) / 32, 32>>>(PARAM_NAMES(__VA_ARGS__)); \
        return 0; \
    } \
    __device__ inline void name ## _impl(const GPUKernelInfo &info, sharedMemoryT &shm, PARAM_LIST(__VA_ARGS__))
