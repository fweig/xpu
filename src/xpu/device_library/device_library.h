#pragma once

template<typename K, typename L>
struct kernel_dispatcher {
    using Library = L;
    using Kernel = K;

    template<typename... Args>
    static inline void dispatch(Library &inst, GPUKernelParams params, Args &&... args) {
        Kernel::dispatch_impl(inst, params, std::forward<Args>(args)...);
    }

    static inline const char *name() {
        return Kernel::name_impl();
    }

};