#ifndef XPU_DETAIL_EXCEPTIONS_H
#define XPU_DETAIL_EXCEPTIONS_H

#include <string_view>

#if __cplusplus >= 202002L // C++20
#define XPU_UNLIKELY_IF(x) if (x) [[unlikely]]
#elif __gcc__ || __clang__
#define XPU_UNLIKELY_IF(x) if (x) __attribute__((__unlikely__))
#else
#define XPU_UNLIKELY_IF(x) if (x)
#endif

#define XPU_CHECK_RANGE(where, i, size) XPU_UNLIKELY_IF((i) >= (size)) xpu::detail::throw_out_of_range(where, (i), (size))

namespace xpu::detail {

[[noreturn]] void throw_out_of_range(std::string_view where, size_t i, size_t size);

} // namespace xpu::detail

#endif
