#include "../../detail/driver_interface.h"
#include "../../detail/log.h"
#include "../../common.h"

#define CONCAT_I(a, b) a ## b
#define CONCAT(a, b) CONCAT_I(a, b)

#if XPU_IS_CUDA
#define CUHIP(expr) CONCAT(cuda, expr)
using cuhip_device_prop = cudaDeviceProp;
#else
#include <hip/hip_runtime_api.h>
#define CUHIP(expr) CONCAT(hip, expr)
using cuhip_device_prop = hipDeviceProp_t;
#endif

namespace xpu {
namespace detail {

class cuhip_driver : public driver_interface {

public:
    virtual ~cuhip_driver() {}

    error setup() override {
        return 0;
    }

    error device_malloc(void **ptr, size_t bytes) override {
        return CUHIP(Malloc)(ptr, bytes);
    }

    error free(void *ptr) override {
        return CUHIP(Free)(ptr);
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
        return CUHIP(GetDeviceCount)(devices);
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
        props->driver = get_type();
        props->major = cuprop.major;
        props->minor = cuprop.minor;

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

};

} // namespace detail
} // namespace xpu

extern "C" xpu::detail::driver_interface *create() {
    return new xpu::detail::cuhip_driver{};
}

extern "C" void destroy(xpu::detail::driver_interface *b) {
    delete b;
}
