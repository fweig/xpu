#define XPU_CMEM_DECL(type, id) \
    extern template struct xpu::cmem_accessor<type>;
