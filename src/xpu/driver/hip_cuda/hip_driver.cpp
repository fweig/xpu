#include <hip/hip_runtime_api.h>

#include <xpu/host.h>

class hip_driver : public xpu::driver_interface {

public:
    xpu::error setup() override {
        xpu::error err;
        
        hipDeviceProp_t props;
        err = hipGetDeviceProperties(&props, 0);
        if (err != 0) {
            return err;
        }

        std::cout << "xpu: selected hip device " << props.name << "(" << props.major << props.minor << ")" << std::endl;

        err = hipSetDevice(0);
        if (err != 0) {
            return err;
        }
        err = hipDeviceSynchronize();

        return err;
    }

    xpu::error device_malloc(void **ptr, size_t bytes) override {
        return hipMalloc(ptr, bytes);
    }

    xpu::error free(void *ptr) override {
        return hipFree(ptr);
    }

    xpu::error memcpy(void *dst, const void *src, size_t bytes) override {
        xpu::error err = hipMemcpy(dst, src, bytes, hipMemcpyDefault);
        hipDeviceSynchronize();
        return err;
    }

    xpu::error memset(void *dst, int ch, size_t bytes) override {
        return hipMemset(dst, ch, bytes);
    }

};

extern "C" xpu::driver_interface *create() {
    return new hip_driver{};
}

extern "C" void destroy(xpu::driver_interface *b) {
    delete b;
}