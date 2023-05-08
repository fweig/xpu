#include "prelude.h"
#include "event.h"
#include "../../backend_base.h"
#include "../../log.h"


namespace xpu::detail {

class CUHIP(driver) : public backend_base {

public:
    virtual ~CUHIP(driver)() {}

    error setup() override {
        error err =
            #if XPU_IS_HIP
                hipInit(0)
            #else
                0
            #endif
        ;
        return err;
    }

    error malloc_device(void **ptr, size_t bytes) override {
        return CUHIP(Malloc)(ptr, bytes);
    }

    error malloc_host(void **ptr, size_t bytes) override {
        return
            #if XPU_IS_HIP
                hipHostMalloc(ptr, bytes, hipHostMallocDefault)
            #else
                cudaHostAlloc(ptr, bytes, 0)
            #endif
        ;
    }

    error malloc_shared(void **ptr, size_t bytes) override {
        return CUHIP(MallocManaged)(ptr, bytes, CUHIP(MemAttachGlobal));
    }

    error free(void *ptr) override {
        if (resides_on_host(ptr)) {
            return
                #if XPU_IS_HIP
                    hipHostFree(ptr)
                #else
                    cudaFreeHost(ptr)
                #endif
            ;
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
        CUHIP(Stream_t) stream;
        err = CUHIP(StreamCreate)(&stream);
        *queue = static_cast<void *>(stream);
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

    error memcpy_async(void *dst, const void *src, size_t bytes, void *queue_handle, double *ms) override {
        CUHIP(Stream_t) queue = static_cast<CUHIP(Stream_t)>(queue_handle);
        if (ms == nullptr) {
            return CUHIP(MemcpyAsync)(dst, src, bytes, CUHIP(MemcpyDefault), queue);
        } else {
            gpu_timer timer;
            timer.start(queue);
            [[maybe_unused]] int err = CUHIP(MemcpyAsync)(dst, src, bytes, CUHIP(MemcpyDefault), queue);
            timer.stop(queue);
            *ms = timer.elapsed();
            return CUHIP(GetLastError)();
        }
    }

    error memset_async(void *dst, int ch, size_t bytes, void *queue_handle, double *ms) override {
        CUHIP(Stream_t) queue = static_cast<CUHIP(Stream_t)>(queue_handle);
        if (ms == nullptr) {
            return CUHIP(MemsetAsync)(dst, ch, bytes, queue);
        } else {
            gpu_timer timer;
            timer.start(queue);
            [[maybe_unused]] int err = CUHIP(MemsetAsync)(dst, ch, bytes, queue);
            timer.stop(queue);
            *ms = timer.elapsed();
            return CUHIP(GetLastError)();
        }
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
        #elif XPU_HIP_VERSION_AT_LEAST(6, 0)
            props->name = "Unknown AMD " + std::string{cuprop.gcnArchName} + " GPU";
        #else // HIP_VERSION < 6.0
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
                *type = mem_host;
                *device = -1;
                return err;
            }

            switch (ptrattrs.type) {
            case cudaMemoryTypeHost:
                *type = mem_pinned;
                *device = ptrattrs.device;
                break;
            case cudaMemoryTypeDevice:
                *type = mem_device;
                *device = ptrattrs.device;
                break;
            case cudaMemoryTypeManaged:
                *type = mem_managed;
                *device = ptrattrs.device;
                break;
            case cudaMemoryTypeUnregistered:
            default:
                *type = mem_host;
                *device = -1;
                break;
            }

        #else // HIP

            if (err == hipErrorInvalidValue) {
                *type = mem_host;
                *device = -1;
                return 0;
            }

            if (err != 0) {
                *type = mem_host;
                *device = -1;
                return err;
            }

            switch (HIP_PTR_TYPE(ptrattrs)) {
            #if XPU_HIP_VERSION_AT_LEAST(6, 0)
                case hipMemoryTypeUnregistered:
                    *type = mem_unknown;
                    *device = -1;
                    break;
            #endif
            case hipMemoryTypeHost:
                *type = mem_pinned;
                *device = ptrattrs.device;
                break;
            case hipMemoryTypeArray:
            case hipMemoryTypeDevice:
                *type = mem_device;
                *device = ptrattrs.device;
                break;
            case hipMemoryTypeManaged:
            case hipMemoryTypeUnified:
                *type = mem_managed;
                *device = ptrattrs.device;
                break;
            }

            #undef PTR_TYPE

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
            return (HIP_PTR_TYPE(ptrattrs) == hipMemoryTypeHost);
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
