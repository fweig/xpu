#define XPU_DRIVER_CPU_DEVICE_RUNTIME

#ifndef XPU_DEVICE_H
#error "This header should not be included directly. Include xpu/device.h instead."
#endif

#include "../../detail/macros.h"
#include "this_thread.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#define XPU_DETAIL_ASSERT(x) assert(x)

namespace xpu {

XPU_FORCE_INLINE int thread_idx::x() { return 0; }
XPU_FORCE_INLINE int block_dim::x() { return 1; }
XPU_FORCE_INLINE int block_idx::x() { return detail::this_thread::block_idx.x; }
XPU_FORCE_INLINE int grid_dim::x() { return detail::this_thread::grid_dim.x; }

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

template<typename Key, typename KeyValueType, int BlockSize, int ItemsPerThread>
class block_sort<Key, BlockSize, ItemsPerThread, driver::cpu> {

public:
    struct storage_t {};

    block_sort(storage_t &) {}

    template<typename KeyGetter>
    KeyValueType *sort(KeyValueType *vals, size_t N, KeyValueType *, KeyGetter &&getKey) {
        std::sort(vals, &vals[N], [&](const KeyValueType &a, const KeyValueType &b) {
            return getKey(a) < getKey(b);
        });
        return vals;
    }

};

} // namespace xpu

#endif
