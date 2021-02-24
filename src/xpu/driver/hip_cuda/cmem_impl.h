#if XPU_IS_CUDA
#define XPU_CMEM_DECL(type, id) \
    xpu::error XPU_DEVICE_LIBRARY::set_cmem_ ## id(const type &symbol) { \
        return cudaMemcpyToSymbol(XPU_CMEM_IDENTIFIER(id), &symbol, sizeof(type)); \
    }
#else
#define XPU_CMEM_DECL(type, id) \
    xpu::error XPU_DEVICE_LIBRARY::set_cmem_ ## id(const type &symbol) { \
        hipMemcpyToSymbol(&XPU_CMEM_IDENTIFIER(id), &symbol, sizeof(type)); \
        hipDeviceSynchronize(); \
        return 0; \
    }
#endif