#ifndef XPU_DEVICE_H
#define XPU_DEVICE_H

#include "defs.h"
#include "host.h"

#define XPU_FE_0(action)
#define XPU_FE_1(action, x, ...) action(x)
#define XPU_FE_2(action, x, ...) action(x),XPU_FE_1(action, __VA_ARGS__)
#define XPU_FE_3(action, x, ...) action(x),XPU_FE_2(action, __VA_ARGS__)
#define XPU_FE_4(action, x, ...) action(x),XPU_FE_3(action, __VA_ARGS__)
#define XPU_FE_5(action, x, ...) action(x),XPU_FE_4(action, __VA_ARGS__)
#define XPU_FE_6(action, x, ...) action(x),XPU_FE_5(action, __VA_ARGS__)
#define XPU_FE_7(action, x, ...) action(x),XPU_FE_6(action, __VA_ARGS__)
#define XPU_FE_8(action, x, ...) action(x),XPU_FE_7(action, __VA_ARGS__)

// FIXME make FOR_EACH macro work with empty arguments
#define XPU_GET_MACRO(_0, _1, _2, _3, _4, _5, _6, _7, _8, NAME, ...) NAME
#define XPU_FOR_EACH(action, ...) \
    XPU_GET_MACRO(_0, __VA_ARGS__, XPU_FE_8, XPU_FE_7, XPU_FE_6, XPU_FE_5, XPU_FE_4, XPU_FE_3, XPU_FE_2, XPU_FE_1, XPU_FE_0)(action, __VA_ARGS__)

#define XPU_EAT(...)
#define XPU_EAT_TYPE(x) XPU_EAT x
#define XPU_STRIP_TYPE_BRACKET_I(...) __VA_ARGS__
#define XPU_STRIP_TYPE_BRACKET(x) XPU_STRIP_TYPE_BRACKET_I x

// Transforms a list of form '(type1) name1, (type2) name2, ...' into 'type1 name1, type2 name2, ...'
#define XPU_PARAM_LIST(...) XPU_FOR_EACH(XPU_STRIP_TYPE_BRACKET, ##__VA_ARGS__)
// Transforms a list of form '(type1) name1, (type2) name2, ...' into 'name1, name2, ...'
#define XPU_PARAM_NAMES(...) XPU_FOR_EACH(XPU_EAT_TYPE, ##__VA_ARGS__)

#define XPU_STRINGIZE_I(val) #val
#define XPU_STRINGIZE(val) XPU_STRINGIZE_I(val)

#define XPU_CONCAT_I(a, b) a##b
#define XPU_CONCAT(a, b) XPU_CONCAT_I(a, b)

#define XPU_IS_HIP_CUDA (XPU_IS_CUDA || XPU_IS_HIP)

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

} // namespace xpu

#if XPU_IS_HIP_CUDA
#include "driver/hip_cuda/device_runtime.h"
#else // CPU
#include "driver/cpu/device_runtime.h"
#endif

#ifndef XPU_KERNEL
#error "XPU_KERNEL not defined."
#endif

#ifndef XPU_ASSERT
#error "XPU_ASSERT not defined."
#endif

namespace xpu {

struct no_smem {};

template<typename C> XPU_D XPU_FORCE_INLINE const C &cmem() { return cmem_accessor<C>::get(); }

XPU_D XPU_FORCE_INLINE constexpr float pi() { return impl::pi(); }
XPU_D XPU_FORCE_INLINE constexpr float deg_to_rad() { return pi() / 180.f; }

XPU_D XPU_FORCE_INLINE int   abs(int x) { return impl::iabs(x); }
XPU_D XPU_FORCE_INLINE float abs(float x) { return impl::fabs(x); }
XPU_D XPU_FORCE_INLINE float ceil(float x) { return impl::ceil(x); }
XPU_D XPU_FORCE_INLINE float cos(float x) { return impl::cos(x); }
XPU_D XPU_FORCE_INLINE int   min(int a, int b) { return impl::imin(a, b); }
XPU_D XPU_FORCE_INLINE float min(float a, float b) { return impl::fmin(a, b); }
XPU_D XPU_FORCE_INLINE int   max(int a, int b) { return impl::imax(a, b); }
XPU_D XPU_FORCE_INLINE float max(float a, float b) { return impl::fmax(a, b); }
XPU_D XPU_FORCE_INLINE float sqrt(float x) { return impl::sqrt(x); }
XPU_D XPU_FORCE_INLINE float tan(float x) { return impl::tan(x); }

XPU_D XPU_FORCE_INLINE int atomic_add_block(int *addr, int val) { return impl::atomic_add_block(addr, val); }


template<typename T, int BlockSize, int ItemsPerThread=8>
class block_sort {

    static_assert(sizeof(T) <= 8, "block_sort can only sort keys with up to 8 bytes...");

public:
    using impl_t = block_sort_impl<T, BlockSize, ItemsPerThread>;
    using storage_t = typename impl_t::storage_t;

    XPU_D XPU_FORCE_INLINE block_sort(storage_t &storage) : impl(storage) {}

    XPU_D XPU_FORCE_INLINE T *sort(T *vals, size_t N, T *buf) { return impl.sort(vals, N, buf); }

private:
    impl_t impl;

};

} // namespace xpu

#endif
