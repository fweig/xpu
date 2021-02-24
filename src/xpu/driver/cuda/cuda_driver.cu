#include <xpu/host.h>

#include <iostream>

class cuda_driver : public xpu::driver_interface {

public:
    xpu::error setup() override{
        std::cout << "xpu: CUDA SETUP" << std::endl;

        xpu::error err;
        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, 0);
        if (err != 0) {
            return err;
        }
    
        std::cout << "xpu: selected cuda device " << props.name << "(" << props.major << props.minor << ")" << std::endl;
    
        err = cudaSetDevice(0);
        if (err != 0) {
            return err;
        }
        err = cudaDeviceSynchronize();
    
        return err;
    }

    xpu::error device_malloc(void **ptr, size_t bytes) override {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        std::cout << "cuda driver: free memory = " << free << "; total = " << total << std::endl;
        std::cout << "cuda driver: allocating " << bytes << " bytes" << std::endl;
        return cudaMalloc(ptr, bytes);
    }

    xpu::error free(void *ptr) override {
        return cudaFree(ptr);
    }

    xpu::error memcpy(void *dst, const void *src, size_t bytes) override {
        xpu::error err = cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
        cudaDeviceSynchronize();
        return err;
    }

    xpu::error memset(void *dst, int ch, size_t bytes) override {
        return cudaMemset(dst, ch, bytes);
    }

};

extern "C" xpu::driver_interface *create() {
    return new cuda_driver{};
}

extern "C" void destroy(xpu::driver_interface *b) {
    delete b;
}