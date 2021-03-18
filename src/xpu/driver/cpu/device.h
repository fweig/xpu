#ifndef XPU_DRIVER_CPU_DEVICE_RUNTIME
#define XPU_DRIVER_CPU_DEVICE_RUNTIME

#ifndef XPU_DEVICE_H
#error "This header should not be included directly. Include xpu/device.h instead."
#endif

#include "../../detail/macros.h"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace xpu {

XPU_FORCE_INLINE int thread_idx::x() { return 0; }
XPU_FORCE_INLINE int block_dim::x() { return 1; }

template<typename S, typename K, typename... Args>
void run_cpu_kernel(int nBlocks, K kernel, Args... args) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int i = 0; i < nBlocks; i++) {
        S smem{};
        xpu::kernel_info info{
            .i_thread = {0, 0, 0},
            .n_threads = {1, 0, 0},
            .i_block = {i, 0, 0},
            .n_blocks = {nBlocks, 0, 0},
        };
        kernel(info, smem, std::forward<Args>(args)...);
    }
}

} // namespace xpu

#define XPU_DETAIL_KERNEL(deviceLibrary, name, sharedMemoryT, ...) \
    void kernel_ ## name(XPU_PARAM_LIST((const xpu::kernel_info &) info, (sharedMemoryT &) smem, ##__VA_ARGS__)); \
    xpu::detail::error deviceLibrary##_Cpu::run_ ## name(XPU_PARAM_LIST((xpu::grid) params, ##__VA_ARGS__)) { \
        if (params.threads.x == -1) { \
            params.threads.x = params.blocks.x; \
        } \
        xpu::run_cpu_kernel<sharedMemoryT>(XPU_PARAM_NAMES(() params.threads.x, () kernel_ ## name, ##__VA_ARGS__)); \
        return 0; \
    } \
    \
    void kernel_ ## name(XPU_PARAM_LIST((__attribute__((unused)) const xpu::kernel_info &) info, (__attribute__((unused)) sharedMemoryT &) shm, ##__VA_ARGS__))

#define XPU_DETAIL_ASSERT(x) assert(x)

namespace xpu {

// math functions
XPU_FORCE_INLINE float ceil(float x) { return std::ceil(x); }
XPU_FORCE_INLINE float cos(float x) { return std::cos(x); }
XPU_FORCE_INLINE float abs(float x) { return std::abs(x); }
XPU_FORCE_INLINE float min(float a, float b) { return std::min(a, b);}
XPU_FORCE_INLINE float max(float a, float b) { return std::max(a, b); }
XPU_FORCE_INLINE int   abs(int a) { return std::abs(a); }
XPU_FORCE_INLINE int   min(int a, int b) { return std::min(a, b); }
XPU_FORCE_INLINE unsigned long long int min(unsigned long long int a, unsigned long long int b) { return std::min(a, b); }
XPU_FORCE_INLINE long long int min(long long int a, long long int b) { return std::min(a, b); }
XPU_FORCE_INLINE int   max(int a, int b) { return std::max(a, b); }
XPU_FORCE_INLINE float sqrt(float x) { return std::sqrt(x); }
XPU_FORCE_INLINE float tan(float x) { return std::tan(x); }

inline int atomic_add_block(int *addr, int val) {
    int old = *addr;
    *addr += val;
    return old;
}

// internal class used to handle constant memory
template<typename C>
struct cmem_accessor {

    static C symbol;

    static C &get() { return symbol; }
};

template<typename C>
C cmem_accessor<C>::symbol;

template<typename C> XPU_FORCE_INLINE const C &cmem() { return cmem_accessor<C>::get(); }

template<typename T, size_t N = sizeof(T)>
struct compare_lower_4_byte {};

template<typename T>
struct compare_lower_4_byte<T, 4> {
    inline bool operator()(T a, T b) {
        return a < b;
    }
};

template<typename T>
struct compare_lower_4_byte<T, 8> {
    union as_llu {
        T val;
        unsigned long long int llu;
    };

    inline bool operator()(T a, T b) {
        as_llu a_{.val = a};
        as_llu b_{.val = b};
        return (a_.llu & 0xFFFFFFFFul) < (b_.llu & 0xFFFFFFFFul);
    }
};

template<typename Key, int BlockSize, int ItemsPerThread>
class block_sort<Key, BlockSize, ItemsPerThread, xpu::driver::cpu> {

public:
    struct storage_t {};

    block_sort(storage_t &) {}

    template<typename T, typename KeyGetter>
    T *sort(T *vals, size_t N, T *, KeyGetter &&getKey) {
        std::sort(vals, &vals[N], [&](const T &a, const T &b) {
            return getKey(a) < getKey(b);
        });
        return vals;
    }

};

} // namespace xpu

#endif
