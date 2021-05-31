#ifndef XPU_DEVICE_H
#define XPU_DEVICE_H

#include "defines.h"
#include "common.h"

#define _USE_MATH_DEFINES
#include <cmath>

#define XPU_IMAGE(image) XPU_DETAIL_IMAGE(image)

#define XPU_EXPORT_CONSTANT(image, type_, name) XPU_DETAIL_EXPORT_CONSTANT(image, type_, name)
#define XPU_EXPORT_FUNC(image, name, ...) XPU_DETAIL_EXPORT_FUNC(image, name, ##__VA_ARGS__)
#define XPU_EXPORT_KERNEL(image, name, ...) XPU_DETAIL_EXPORT_KERNEL(image, name, ##__VA_ARGS__)

#define XPU_CONSTANT(name) XPU_DETAIL_CONSTANT(name)
#define XPU_FUNC(name, ...) XPU_DETAIL_FUNC(name, ##__VA_ARGS__)
#define XPU_FUNC_T(name, ...) XPU_DETAIL_FUNC_T(name, ##__VA_ARGS__)
#define XPU_FUNC_TI(name) XPU_DETAIL_FUNC_TI(name)
#define XPU_FUNC_TS(name, ...) XPU_DETAIL_FUNC_TS(name, ##__VA_ARGS__)
#define XPU_KERNEL(name, shared_memory, ...) XPU_DETAIL_KERNEL(name, shared_memory, ##__VA_ARGS__)

#define XPU_ASSERT(x) XPU_DETAIL_ASSERT(x)

#define XPU_T(...) __VA_ARGS__

namespace xpu {

struct thread_idx {
    thread_idx() = delete;
    XPU_D static XPU_FORCE_INLINE int x();
};

struct block_dim {
    block_dim() = delete;
    XPU_D static XPU_FORCE_INLINE int x();
};

struct block_idx {
    block_idx() = delete;
    XPU_D static XPU_FORCE_INLINE int x();
};

struct grid_dim {
    grid_dim() = delete;
    XPU_D static XPU_FORCE_INLINE int x();
};

struct no_smem {};

template<typename C>
XPU_D XPU_FORCE_INLINE const typename C::data_t &cmem() { return C::get(); }

XPU_D XPU_FORCE_INLINE constexpr float pi() { return M_PIf32; }
XPU_D XPU_FORCE_INLINE constexpr float deg_to_rad() { return pi() / 180.f; }

XPU_D XPU_FORCE_INLINE int   abs(int x);
XPU_D XPU_FORCE_INLINE float abs(float x);
XPU_D XPU_FORCE_INLINE float ceil(float x);
XPU_D XPU_FORCE_INLINE float cos(float x);
XPU_D XPU_FORCE_INLINE int   min(int a, int b);
XPU_D XPU_FORCE_INLINE unsigned long long int min(unsigned long long int a, unsigned long long int b);
XPU_D XPU_FORCE_INLINE long long int min(long long int a, long long int b);
XPU_D XPU_FORCE_INLINE float min(float a, float b);
XPU_D XPU_FORCE_INLINE int   max(int a, int b);
XPU_D XPU_FORCE_INLINE float max(float a, float b);
XPU_D XPU_FORCE_INLINE float sqrt(float x);
XPU_D XPU_FORCE_INLINE float tan(float x);

XPU_D XPU_FORCE_INLINE int atomic_add_block(int *addr, int val);

template<typename Key, typename KeyValueType, int BlockSize, int ItemsPerThread=8, xpu::driver Impl=XPU_COMPILATION_TARGET>
class block_sort {

public:
    struct storage_t {};

    XPU_D block_sort(storage_t &);

    template<typename KeyGetter>
    XPU_D KeyValueType *sort(KeyValueType *vals, size_t N, KeyValueType *buf, KeyGetter &&getKey);
};

template<typename Key, int BlockSize, int ItemsPerThread=8, xpu::driver Impl=XPU_COMPILATION_TARGET>
class block_merge {

public:
    struct storage_t {};

    XPU_D block_merge(storage_t &);

    template<typename Compare>
    XPU_D void merge(const Key *a, size_t size_a, const Key *b, size_t size_b, Key *dst, Compare &&);

};

} // namespace xpu

#include "detail/dynamic_loader.h"

#if XPU_IS_HIP_CUDA
#include "driver/hip_cuda/device.h"
#else // CPU
#include "driver/cpu/device.h"
#endif

#endif
