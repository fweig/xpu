#include "../../backend_base.h"
#include "../../log.h"
#include "../../macros.h"
#include "../../../defines.h"

#if XPU_IS_CUDA
#define CUHIP(expr) XPU_CONCAT(cuda, expr)
using cuhip_device_prop = cudaDeviceProp;
using cuhip_pointer_attributes = cudaPointerAttributes;
#else
#include <hip/hip_runtime_api.h>
#define CUHIP(expr) XPU_CONCAT(hip, expr)
using cuhip_device_prop = hipDeviceProp_t;
using cuhip_pointer_attributes = hipPointerAttribute_t;
#endif

namespace xpu::detail {

class CUHIP(driver) : public backend_base {

public:
    virtual ~CUHIP(driver)() {}

    error setup() override {
        return 0;
    }

    error malloc_device(void **ptr, size_t bytes) override {
        return CUHIP(Malloc)(ptr, bytes);
    }

    error malloc_host(void **ptr, size_t bytes) override {
        return CUHIP(MallocHost)(ptr, bytes);
    }

    error malloc_shared(void **ptr, size_t bytes) override {
        return CUHIP(MallocManaged)(ptr, bytes, CUHIP(MemAttachGlobal));
    }

    error free(void *ptr) override {
        if (resides_on_host(ptr)) {
            return CUHIP(FreeHost)(ptr);
        } else {
            return CUHIP(Free)(ptr);
        }
    }

    error create_queue(void **queue, int device) override {
        int err = 0;
        int current_device = 0;
        err = CUHIP(GetDevice)(&current_device);
        if (err != 0) {
            return err;
        }
        err = set_device(device);
        if (err != 0) {
            return err;
        }
        CUHIP(Stream_t) *stream = static_cast<CUHIP(Stream_t) *>(*queue);
        err = CUHIP(StreamCreate)(stream);
        if (err != 0) {
            return err;
        }
        err = CUHIP(SetDevice)(current_device);
        return err;
    }

    error destroy_queue(void *queue) override {
        return CUHIP(StreamDestroy)(static_cast<CUHIP(Stream_t)>(queue));
    }

    error synchronize_queue(void *queue) override {
        return CUHIP(StreamSynchronize)(static_cast<CUHIP(Stream_t)>(queue));
    }

    error memcpy(void *dst, const void *src, size_t bytes) override {
        error err = CUHIP(Memcpy)(dst, src, bytes, CUHIP(MemcpyDefault));
        device_synchronize();
        return err;
    }

    error memset(void *dst, int ch, size_t bytes) override {
        return CUHIP(Memset)(dst, ch, bytes);
    }

    error num_devices(int *devices) override {
        CUHIP(GetDeviceCount)(devices);
        return 0;
    }

    error set_device(int device) override {
        return CUHIP(SetDevice)(device);
    }

    error get_device(int *device) override {
        return CUHIP(GetDevice)(device);
    }

    error device_synchronize() override {
        return CUHIP(DeviceSynchronize)();
    }

    error get_properties(device_prop *props, int device) override {
        cuhip_device_prop cuprop;
        error err = CUHIP(GetDeviceProperties)(&cuprop, device);
        if (err != 0) {
            return err;
        }

        props->name = cuprop.name;

        // Name field (which returns marketing name) on HIP is bugged
        // and returns empty string for some devices.
        // So we need to manually set a fallback name.
        if (props->name.empty()) {
        #if XPU_IS_CUDA
            props->name = "Unknown NVIDIA GPU";
        #else // XPU_IS_HIP
            props->name = "Unknown AMD gfx" + std::to_string(cuprop.gcnArch) + " GPU";
        #endif
        }

        props->driver = get_type();
        props->arch = std::to_string(cuprop.major) + std::to_string(cuprop.minor);

        props->shared_mem_size = cuprop.sharedMemPerBlock;
        props->const_mem_size = cuprop.totalConstMem;

        props->warp_size = cuprop.warpSize;
        props->max_threads_per_block = cuprop.maxThreadsPerBlock;
        props->max_grid_size = {size_t(cuprop.maxGridSize[0]), size_t(cuprop.maxGridSize[1]), size_t(cuprop.maxGridSize[2])};

        return 0;
    }

    error get_ptr_prop(const void *ptr, int *device, mem_type *type) override {
        cuhip_pointer_attributes ptrattrs;
        error err = CUHIP(PointerGetAttributes)(&ptrattrs, ptr);

        #if XPU_IS_CUDA
            if (err != 0) {
                *type = mem_unknown;
                *device = -1;
                return err;
            }

            switch (ptrattrs.type) {
            case cudaMemoryTypeHost:
                *type = mem_host;
                *device = ptrattrs.device;
                break;
            case cudaMemoryTypeDevice:
                *type = mem_device;
                *device = ptrattrs.device;
                break;
            case cudaMemoryTypeManaged:
                *type = mem_shared;
                *device = ptrattrs.device;
                break;
            case cudaMemoryTypeUnregistered:
            default:
                *type = mem_unknown;
                *device = -1;
                break;
            }

        #else // HIP

            if (err == hipErrorInvalidValue) {
                *type = mem_unknown;
                *device = -1;
                return 0;
            }

            if (err != 0) {
                *type = mem_unknown;
                *device = -1;
                return err;
            }

            switch (ptrattrs.memoryType) {
            case hipMemoryTypeHost:
                *type = mem_host;
                *device = ptrattrs.device;
                break;
            case hipMemoryTypeArray:
            case hipMemoryTypeDevice:
                *type = mem_device;
                *device = ptrattrs.device;
                break;
            case hipMemoryTypeManaged:
                *type = mem_shared;
                *device = ptrattrs.device;
                break;
            }

        #endif

        return 0;
    }

    error meminfo(size_t *free, size_t *total) override {
        return CUHIP(MemGetInfo)(free, total);
    }

    const char *error_to_string(error err) override {
        return CUHIP(GetErrorString)(static_cast<CUHIP(Error_t)>(err));
    }

    driver_t get_type() override {
        return (XPU_IS_CUDA ? cuda : hip);
    }

    bool resides_on_host(const void *ptr) {
        cuhip_pointer_attributes ptrattrs;
        error err = CUHIP(PointerGetAttributes)(&ptrattrs, ptr);
        if (err != 0) {
            return false;
        }

        #if XPU_IS_CUDA
            return (ptrattrs.type == cudaMemoryTypeHost);
        #else
            return (ptrattrs.memoryType == hipMemoryTypeHost);
        #endif
    }

};

} // namespace xpu::detail

extern "C" xpu::detail::backend_base *create() {
    return new xpu::detail::CUHIP(driver){};
}

extern "C" void destroy(xpu::detail::backend_base *b) {
    delete b;
}
