#define XPU_CMEM_DECL(type, id) \
    template struct xpu::cmem_accessor<type>; \
    xpu::detail::error XPU_DEVICE_LIBRARY::set_cmem_ ## id(const type &symbol) { \
        xpu::cmem_accessor<type>::get() = symbol; \
        return 0; \
    }
