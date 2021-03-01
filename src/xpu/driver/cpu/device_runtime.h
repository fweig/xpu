#ifndef XPU_DRIVER_CPU_DEVICE_RUNTIME
#define XPU_DRIVER_CPU_DEVICE_RUNTIME

#include <algorithm>
#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>

namespace xpu {

inline int thread_idx::x() { return 0; }
inline int block_dim::x() { return 1; }

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

#define XPU_KERNEL(deviceLibrary, name, sharedMemoryT, ...) \
    void kernel_ ## name(XPU_PARAM_LIST((const xpu::kernel_info &) info, (sharedMemoryT &) smem, ##__VA_ARGS__)); \
    xpu::error deviceLibrary##_Cpu::run_ ## name(XPU_PARAM_LIST((xpu::grid) params, ##__VA_ARGS__)) { \
        if (params.threads.x == -1) { \
            params.threads.x = params.blocks.x; \
        } \
        xpu::run_cpu_kernel<sharedMemoryT>(XPU_PARAM_NAMES(() params.threads.x, () kernel_ ## name, ##__VA_ARGS__)); \
        return 0; \
    } \
    \
    void kernel_ ## name(XPU_PARAM_LIST((__attribute__((unused)) const xpu::kernel_info &) info, (__attribute__((unused)) sharedMemoryT &) shm, ##__VA_ARGS__))

#define XPU_ASSERT(x) assert(x)

namespace xpu {

// math functions
namespace impl {
constexpr float pi() { return static_cast<float>(M_PI); }

inline float ceil(float x) { return std::ceil(x); }
inline float cos(float x) { return std::cos(x); }
inline float fabs(float x) { return std::abs(x); }
inline float fmin(float a, float b) { return std::min(a, b);}
inline float fmax(float a, float b) { return std::max(a, b); }
inline int   iabs(int a) { return std::abs(a); }
inline int   imin(int a, int b) { return std::min(a, b); }
inline int   imax(int a, int b) { return std::max(a, b); }
inline float sqrt(float x) { return std::sqrt(x); }
inline float tan(float x) { return std::tan(x); }

inline int atomic_add_block(int *addr, int val) {
    int old = *addr;
    *addr += val;
    return old;
}

} // namespace impl

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

template<typename T, int BlockSize, int ItemsPerThread>
class block_sort_impl {

public:
    struct storage_t {};

    block_sort_impl(storage_t &) {}

    T *sort(T *vals, size_t N, T *) {
        compare_lower_4_byte<T> comp{};
        std::sort(vals, &vals[N], [&](T a, T b) {
            return comp(a, b);
        });
        return vals;
    }

};

// internal class used to handle constant memory
template<typename C>
struct cmem_accessor {

    static C symbol;

    static C &get() { return symbol; }
};

template<typename C>
C cmem_accessor<C>::symbol;

} // namespace xpu

#endif
