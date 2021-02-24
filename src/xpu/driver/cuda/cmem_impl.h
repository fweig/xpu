#define XPU_CMEM_DECL(type, id) \
    xpu::error XPU_DEVICE_LIBRARY::set_cmem_ ## id(const type &symbol) { \
        return cudaMemcpyToSymbol(XPU_CMEM_IDENTIFIER(id), &symbol, sizeof(type)); \
    }
