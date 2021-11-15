#ifndef XPU_DRIVER_CPU_DEVICE_RUNTIME_H
#define XPU_DRIVER_CPU_DEVICE_RUNTIME_H

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

namespace detail {

// workaround until c++14 / c++17 with std::exchange is available
template<class T>
inline T exchange(T &obj, T new_val) {
    T old_val = std::move(obj);
    obj = new_val;
    return old_val;
}

} // namespace detail

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
XPU_FORCE_INLINE unsigned int min(unsigned int a, unsigned int b) { return std::min(a, b); }
XPU_FORCE_INLINE unsigned long long int min(unsigned long long int a, unsigned long long int b) { return std::min(a, b); }
XPU_FORCE_INLINE long long int min(long long int a, long long int b) { return std::min(a, b); }
XPU_FORCE_INLINE int   max(int a, int b) { return std::max(a, b); }
XPU_FORCE_INLINE float sqrt(float x) { return std::sqrt(x); }
XPU_FORCE_INLINE float tan(float x) { return std::tan(x); }

inline int atomic_cas(int *addr, int compare, int val) {
    __atomic_compare_exchange(addr, &compare, &val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return compare;
}

inline unsigned int atomic_cas(unsigned int *addr, unsigned int compare, unsigned int val) {
    __atomic_compare_exchange(addr, &compare, &val, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return compare;
}

inline int atomic_cas_block(int *addr, int compare, int val) {
    return detail::exchange(*addr, (*addr == compare ? val : *addr));
}

inline unsigned int atomic_cas_block(unsigned int *addr, unsigned int compare, unsigned int val) {
    return detail::exchange(*addr, (*addr == compare ? val : *addr));
}

inline int atomic_add(int *addr, int val) {
    return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int atomic_add(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
}

inline int atomic_add_block(int *addr, int val) {
    return detail::exchange(*addr, *addr + val);
}

inline unsigned int atomic_add_block(unsigned int *addr, unsigned int val) {
    return detail::exchange(*addr, *addr + val);
}

inline int atomic_sub(int *addr, int val) {
    return __atomic_fetch_sub(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int atomic_sub(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_sub(addr, val, __ATOMIC_SEQ_CST);
}

inline int atomic_sub_block(int *addr, int val) {
    return detail::exchange(*addr, *addr - val);
}

inline unsigned int atomic_sub_block(unsigned int *addr, unsigned int val) {
    return detail::exchange(*addr, *addr - val);
}

inline int atomic_and(int *addr, int val) {
    return __atomic_fetch_and(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int atomic_and(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_and(addr, val, __ATOMIC_SEQ_CST);
}

inline int atomic_and_block(int *addr, int val) {
    return detail::exchange(*addr, *addr & val);
}

inline unsigned int atomic_and_block(unsigned int *addr, unsigned int val) {
    return detail::exchange(*addr, *addr & val);
}

inline int atomic_or(int *addr, int val) {
    return __atomic_fetch_or(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int atomic_or(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_or(addr, val, __ATOMIC_SEQ_CST);
}

inline int atomic_or_block(int *addr, int val) {
    return detail::exchange(*addr, *addr | val);
}

inline unsigned int atomic_or_block(unsigned int *addr, unsigned int val) {
    return detail::exchange(*addr, *addr | val);
}

inline int atomic_xor(int *addr, int val) {
    return __atomic_fetch_xor(addr, val, __ATOMIC_SEQ_CST);
}

inline unsigned int atomic_xor(unsigned int *addr, unsigned int val) {
    return __atomic_fetch_xor(addr, val, __ATOMIC_SEQ_CST);
}

inline int atomic_xor_block(int *addr, int val) {
    return detail::exchange(*addr, *addr ^ val);
}

inline unsigned int atomic_xor_block(unsigned int *addr, unsigned int val) {
    return detail::exchange(*addr, *addr ^ val);
}

XPU_FORCE_INLINE void barrier() { return; }

template<typename Key, typename KeyValueType, int BlockSize, int ItemsPerThread>
class block_sort<Key, KeyValueType, BlockSize, ItemsPerThread, cpu> {

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

template<typename Key, int BlockSize, int ItemsPerThread>
class block_merge<Key, BlockSize, ItemsPerThread, cpu> {

public:
    struct storage_t {};

    block_merge(storage_t &) {}

    template<typename Compare>
    void merge(const Key *a, size_t size_a, const Key *b, size_t size_b, Key *dst, Compare &&comp) {
        std::merge(a, a + size_a, b, b + size_b, dst, comp);
    }

};

} // namespace xpu

#endif
