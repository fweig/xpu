#pragma once

#include "../host.h"

namespace xpu {

template<typename K, typename L>
struct kernel_dispatcher {
    using library = L;
    using kernel = K;

    template<typename... Args>
    static inline void dispatch(library &inst, grid params, Args &&... args) {
        kernel::dispatch_impl(inst, params, std::forward<Args>(args)...);
    }

    static inline const char *name() {
        return kernel::name_impl();
    }
};

}