#ifndef XPU_DRIVER_CPU_BLOCK_OPS_H
#define XPU_DRIVER_CPU_BLOCK_OPS_H

#include <algorithm>

namespace xpu {

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

} // namespace xpu

#endif
