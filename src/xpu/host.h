#pragma once

#include "defs.h"

#include <cstddef>
#include <iostream>
#include <utility>
#include <type_traits>

namespace xpu {

enum class driver {
    cpu,
    cuda,
};


template<class K>
struct is_kernel : std::false_type {};

struct lane {
    struct standard_t {};
    static constexpr standard_t standard{};

    struct cpu_t {};
    static constexpr cpu_t cpu{};

    int value;

    lane(standard_t) : value(0) {}
    lane(cpu_t) : value(0) {}
};

struct dim {
    int x = 0; 
    int y = 0; 
    int z = 0;

    XPU_D dim(int x) : x(x) {}
    XPU_D dim(int x, int y) : x(x), y(y) {}
    XPU_D dim(int x, int y, int z) : x(x), y(y), z(z) {}
};

struct grid {

    static inline grid n_blocks(dim blocks, lane l = lane::standard) {
        return grid{blocks, dim{-1}, l};
    }

    static inline grid n_threads(dim threads, lane l = lane::standard) {
        return grid{dim{-1}, threads, l};
    }

    static inline grid fill(lane l = lane::standard);

    dim blocks;
    dim threads;

private:
    grid(dim b, dim t, lane) : blocks(b), threads(t) {} 

};

void initialize(driver);
void *device_malloc(size_t);

template<typename T>
T *device_malloc(size_t N) {
    return static_cast<T *>(device_malloc(sizeof(T) * N));
}

void free(void *);
void memcpy(void *, const void *, size_t);

template<typename T>
void copy(T *dst, const T *src, size_t entries) {
    memcpy(dst, src, sizeof(T) * entries);
}

driver active_driver();

template<typename Kernel, typename Enable = typename std::enable_if<is_kernel<Kernel>::value>::type>
const char *get_name() {
    return Kernel::name();
}

template<typename Kernel, typename Enable = typename std::enable_if<is_kernel<Kernel>::value>::type, typename... Args>
void run_kernel(grid params, Args&&... args) {
    std::string backend = "CPU";
    if (active_driver() == driver::cuda) {
        backend = "CUDA";
    }
    std::cout << "Running kernel " << get_name<Kernel>() << " on backend " << backend << std::endl;
    Kernel::dispatch(Kernel::library::instance(active_driver()), params, std::forward<Args>(args)...);
}

}