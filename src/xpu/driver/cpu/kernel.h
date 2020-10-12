#pragma once

#include "../../macros.h"

#define XPU_KERNEL(name, sharedMemoryT, ...) \
    void kernel_ ## name(const xpu::kernel_info &, sharedMemoryT &, PARAM_LIST(__VA_ARGS__)); \
    xpu::error XPU_DEVICE_LIBRARY_BACKEND_NAME::run_ ## name(xpu::grid params, PARAM_LIST(__VA_ARGS__)) { \
        for (int i = 0; i < params.threads.x; i++) { \
            sharedMemoryT shm{}; \
            xpu::kernel_info info{ \
            .i_thread = {0, 0, 0}, \
            .n_threads  = {1, 0, 0}, \
            .i_block  = {i, 0, 0}, \
            .n_blocks   = {params.threads.x, 0, 0}}; \
            \
            kernel_ ## name(info, shm, PARAM_NAMES(__VA_ARGS__)); \
        } \
        return 0; \
    } \
    \
    void kernel_ ## name(const xpu::kernel_info &info, sharedMemoryT &shm, PARAM_LIST(__VA_ARGS__))
