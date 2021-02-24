#if XPU_IS_HIP_CUDA
#define XPU_CMEM_DECL(type, id) \
    __constant__ type XPU_CMEM_IDENTIFIER(id); \
    template<> \
    struct xpu::cmem_accessor<type> { \
        __device__ static XPU_INLINE const type &get() { \
            return XPU_CMEM_IDENTIFIER(id); \
        } \
    };
#else
#define XPU_CMEM_DECL(...)
#endif