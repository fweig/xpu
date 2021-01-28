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

#define XPU_GET_MACRO(_0, _1, _2, _3, _4, _5, _6, _7, _8, NAME, ...) NAME
#define XPU_FOR_EACH(action, ...) \
    XPU_GET_MACRO(_0, __VA_ARGS__, XPU_FE_8, XPU_FE_7, XPU_FE_6, XPU_FE_5, XPU_FE_4, XPU_FE_3, XPU_FE_2, XPU_FE_1, XPU_FE_0)(action, __VA_ARGS__)

#define XPU_EAT(...)
#define XPU_EAT_TYPE(x) XPU_EAT x
#define XPU_STRIP_TYPE_BRACKET_I(...) __VA_ARGS__
#define XPU_STRIP_TYPE_BRACKET(x) XPU_STRIP_TYPE_BRACKET_I x

#define XPU_PARAM_LIST(...) XPU_FOR_EACH(XPU_STRIP_TYPE_BRACKET, __VA_ARGS__)
#define XPU_PARAM_NAMES(...) XPU_FOR_EACH(XPU_EAT_TYPE, __VA_ARGS__)

#define XPU_STRINGIZE_I(val) #val
#define XPU_STRINGIZE(val) XPU_STRINGIZE_I(val)

#if defined(__NVCC__)
#include "driver/cuda/device_runtime.h"
#else // CPU
#include "driver/cpu/device_runtime.h"
#endif

#ifndef XPU_KERNEL
#error "XPU_KERNEL not defined."
#endif

namespace xpu {

struct no_smem {};

template<typename T, int BlockSize>
class block_sort {

public:
    using impl_t = block_sort_impl<T, BlockSize>;
    using storage_t = typename impl_t::storage;

    XPU_DI block_sort(storage_t &storage) : impl(storage) {}

    XPU_DI void sort(T *vals, size_t N) {
        impl.sort(vals, N);
    }

private:
    impl_t impl;

};

} // namespace xpu

#endif