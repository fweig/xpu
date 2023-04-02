#ifndef XPU_DETAIL_BACKEND_H
#define XPU_DETAIL_BACKEND_H

#include "backend_base.h"
#include "common.h"

namespace xpu::detail::backend {

void load();
bool is_available(driver_t);
backend_base *get(driver_t, bool = true);
[[noreturn]] void raise_error(driver_t, error);

template<typename... Args1, typename... Args2>
void call(driver_t driver, error (backend_base::*func)(Args1...), Args2&&... args) {
    auto backend = get(driver);
    error err = (backend->*func)(std::forward<Args2>(args)...);
    if (err != 0) {
        raise_error(driver, err);
    }
}

} // namespace xpu::detail::backend

#endif
