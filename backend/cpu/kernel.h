#pragma once

#define FE_0(WHAT) 
#define FE_1(WHAT, x, ...) WHAT(x)
#define FE_2(WHAT, x, ...) WHAT(x)FE_1(WHAT, __VA_ARGS__)
#define FE_3(WHAT, x, ...) WHAT(x)FE_2(WHAT, __VA_ARGS__)
#define FE_4(WHAT, x, ...) WHAT(x)FE_3(WHAT, __VA_ARGS__)
#define FE_5(WHAT, x, ...) WHAT(x)FE_4(WHAT, __VA_ARGS__)

#define GET_MACRO(_0, _1, _2, _3, _4, _5, NAME, ...) NAME
#define FOR_EACH(action, ...) \
    GET_MACRO(_0, __VA_ARGS__, FE_5, FE_4, FE_3, FE_2, FE_2, FE_0)(action, __VA_ARGS__)

#define EAT(...)
#define EAT_TYPE(x) , EAT x
#define ID(x) x
#define STRIP_TYPE_BRACKET(x) , ID x

#define PARAM_LIST(...) FOR_EACH(STRIP_TYPE_BRACKET, __VA_ARGS__)
#define PARAM_NAMES(...) FOR_EACH(EAT_TYPE, __VA_ARGS__)

#define KERNEL_IMPL(name, sharedMemoryT, ...) \
    void kernel_ ## name(const GPUKernelInfo &, sharedMemoryT & PARAM_LIST(__VA_ARGS__)); \
    GPUError CPUTestKernels::name(gpu::internal::KernelTag, GPUKernelParams params PARAM_LIST(__VA_ARGS__)) { \
        for (int i = 0; i < params.range.x; i++) { \
            sharedMemoryT shm{}; \
            GPUKernelInfo info{ \
            .threadIdx = {0, 0, 0}, \
            .nThreads  = {1, 0, 0}, \
            .blockIdx  = {i, 0, 0}, \
            .nBlocks   = {params.range.x, 0, 0}}; \
            \
            kernel_ ## name(info, shm PARAM_NAMES(__VA_ARGS__)); \
        } \
        return 0; \
    } \
    \
    void kernel_ ## name(const GPUKernelInfo &info, sharedMemoryT &shm PARAM_LIST(__VA_ARGS__))
