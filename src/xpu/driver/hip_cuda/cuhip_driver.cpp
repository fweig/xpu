#include "../../detail/driver_interface.h"
#include "../../detail/log.h"
#include "../../detail/macros.h"
#include "../../common.h"

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

class CUHIP(driver) : public driver_interface {

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
        props->major = cuprop.major;
        props->minor = cuprop.minor;

        return 0;
    }

    error pointer_get_device(const void *ptr, int *device) override {
        cuhip_pointer_attributes ptrattrs;
        error err = CUHIP(PointerGetAttributes)(&ptrattrs, ptr);

        #if XPU_IS_CUDA
            if (err != 0) {
                return err;
            }

            if (ptrattrs.type == cudaMemoryTypeUnregistered || ptrattrs.type == cudaMemoryTypeHost) {
                *device = -1;
            } else {
                *device = ptrattrs.device;
            }

        #else // HIP
            if (err == hipErrorInvalidValue) {
                *device = -1;
                return 0;
            }

            if (err != 0) {
                return err;
            }

            if (ptrattrs.memoryType == hipMemoryTypeHost) {
                *device = -1;
                return 0;
            }

            *device = ptrattrs.device;

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

extern "C" xpu::detail::driver_interface *create() {
    return new xpu::detail::CUHIP(driver){};
}

extern "C" void destroy(xpu::detail::driver_interface *b) {
    delete b;
}
