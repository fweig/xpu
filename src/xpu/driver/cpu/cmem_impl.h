#define XPU_CMEM_DECL(type, id) \
    template struct xpu::cmem_accessor<type>; \
    xpu::error BackendT::set_cmem_ ## id(const type &symbol) { \
        xpu::cmem_accessor<type>::get() = symbol; \
        return 0; \
    }
