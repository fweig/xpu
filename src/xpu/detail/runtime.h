#ifndef XPU_DETAIL_RUNTIME_H
#define XPU_DETAIL_RUNTIME_H

#include "../common.h"
#include "../driver/cpu/cpu_driver.h"
#include "dl_utils.h"
#include "dynamic_loader.h"
#include "log.h"

#include <memory>

namespace xpu {
namespace detail {

class image_pool {

public:
    template<typename I>
    I *find(driver_t d) {
        for (auto &e : entries) {
            if (e.d == d && e.id == type_id<I, image_pool>::get()) {
                return static_cast<I *>(e.image);
            }
        }
        return nullptr;
    }

    template<typename I>
    void add(I *i, driver_t d) {
        entries.push_back(basic_entry{d, type_id<I, image_pool>::get(), i});
    }

private:
    struct basic_entry {
        driver_t d;
        size_t id;
        void *image;
    };

    std::vector<basic_entry> entries;
};

class runtime {

public:
    static runtime &instance();

    void initialize();

    void *host_malloc(size_t);
    void *device_malloc(size_t);

    void free(void *);
    void memcpy(void *, const void *, size_t);
    void memset(void *, int, size_t);

    device_prop device_properties();

    driver_t active_driver() const { return _active_driver; }

    template<typename Kernel, typename... Args>
    void run_kernel(grid g, Args&&... args) {
        float ms;
        get_image<Kernel>()->template run_kernel<Kernel>((measure_time ? &ms : nullptr), g, std::forward<Args>(args)...);

        if (measure_time) {
            size_t id = type_id<Kernel, typename Kernel::image>::get();

            if (profiling.size() <= id) {
                profiling.resize(id+1);
            }

            profiling.at(id).push_back(ms);
        }
    }

    template<typename C>
    void set_constant(const typename C::data_t &symbol) {
        get_image<C>()->template set<C>(symbol);
    }

    template<typename Kernel>
    std::vector<float> get_timing() {
        size_t id = type_id<Kernel, typename Kernel::image>::get();

        // Profiling not enabled or kernel hasn't run yet.
        if (profiling.size() <= id) {
            return {};
        }

        return profiling.at(id);
    }

private:
    std::unique_ptr<cpu_driver> the_cpu_driver;
    std::unique_ptr<lib_obj<driver_interface>> the_cuda_driver;
    std::unique_ptr<lib_obj<driver_interface>> the_hip_driver;

    driver_t _active_driver = cpu;

    image_pool images;

    bool measure_time = false;
    std::vector<std::vector<float>> profiling;

    std::vector<device_prop> devices;

    template<typename A>
    image<typename A::image> *get_image() {
        auto *img = images.find< image<typename A::image> >(active_driver());
        if (img == nullptr) {
            img = load_image<typename A::image>(active_driver());
        }
        if (img == nullptr) {
            XPU_LOG("Failed to load image for kernel '%s'.", type_name<A>());
            std::abort();
        }
        return img;
    }

    template<typename I>
    image<I> *load_image(driver_t d) {
        image<I> *i = nullptr;
        switch (d) {
        case cpu:
            i = new image<I>{};
            break;
        case cuda:
        case hip:
            i = new image<I>(complete_file_name(image_context<I>::file_name, d).c_str());
            break;
        }
        images.add(i, d);
        return i;
    }

    bool has_driver(driver_t) const;
    driver_interface *get_driver(driver_t) const;
    driver_interface *get_active_driver() const;

    std::string complete_file_name(const char *, driver_t) const;

    const char *driver_str(driver_t) const;

    void throw_on_driver_error(driver_t, error) const;
};

} // namespace detail
} // namespace xpu

#endif
