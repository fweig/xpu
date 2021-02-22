#ifndef XPU_DRIVER_CPU_DEVICE_RUNTIME
#define XPU_DRIVER_CPU_DEVICE_RUNTIME

#include <algorithm>
#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>

namespace xpu {

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

#define XPU_KERNEL(name, sharedMemoryT, ...) \
    void kernel_ ## name(XPU_PARAM_LIST((const xpu::kernel_info &) info, (sharedMemoryT &) smem, ##__VA_ARGS__)); \
    xpu::error XPU_CONCAT(XPU_DEVICE_LIBRARY, XPU_DRIVER_NAME)::run_ ## name(XPU_PARAM_LIST((xpu::grid) params, ##__VA_ARGS__)) { \
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
} // namespace impl

template<typename T, int BlockSize>
class block_sort_impl {

public:
    struct storage {};

    block_sort_impl(storage &) {}

    void sort(T *vals, size_t N) { std::sort(vals, &vals[N]); }

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