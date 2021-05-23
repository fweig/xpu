#ifndef XPU_DETAIL_RUNTIME_H
#define XPU_DETAIL_RUNTIME_H

#include "../common.h"
#include "../driver/cpu/cpu_driver.h"
#include "dl_utils.h"
#include "dynamic_loader.h"

#include <memory>

namespace xpu {
namespace detail {

class image_pool {

public:
    template<typename I>
    I *find(driver d) {
        for (auto &e : entries) {
            if (e.d == d && e.id == type_id<I, image_pool>::get()) {
                return static_cast<I *>(e.image);
            }
        }
        return nullptr;
    }

    template<typename I>
    void add(I *i, driver d) {
        entries.push_back(basic_entry{d, type_id<I, image_pool>::get(), i});
    }

private:
    struct basic_entry {
        driver d;
        size_t id;
        void *image;
    };

    std::vector<basic_entry> entries;
};

class runtime {

public:
    static runtime &instance();

    void initialize(driver);

    void *host_malloc(size_t);
    void *device_malloc(size_t);

    void free(void *);
    void memcpy(void *, const void *, size_t);
    void memset(void *, int, size_t);

    driver active_driver() { return active_driver_type; }

    template<typename Kernel, typename... Args>
    void run_kernel(grid g, Args&&... args) {
        get_image<Kernel>()->template run_kernel<Kernel>(g, std::forward<Args>(args)...);
    };

    template<typename C>
    void set_constant(const typename C::data_t &symbol) {
        get_image<C>()->template set<C>(symbol);
    }

private:
    std::unique_ptr<cpu_driver> the_cpu_driver;
    std::unique_ptr<lib_obj<driver_interface>> the_cuda_driver;
    std::unique_ptr<lib_obj<driver_interface>> the_hip_driver;
    driver_interface *active_driver_inst = nullptr;
    driver active_driver_type = driver::cpu;

    image_pool images;

    template<typename A>
    image<typename A::image> *get_image() {
        auto *img = images.find< image<typename A::image> >(active_driver_type);
        if (img == nullptr) {
            img = load_image<typename A::image>(active_driver_type);
        }
        if (img == nullptr) {
            std::cout << "Failed to load image for " << type_name<A>() << std::endl;
            std::abort();
        }
        return img;
    }

    template<typename I>
    image<I> *load_image(driver d) {
        image<I> *i = nullptr;
        switch (d) {
        case driver::cpu:
            i = new image<I>{};
            break;
        case driver::cuda:
        case driver::hip:
            i = new image<I>(complete_file_name(image_context<I>::file_name, d).c_str());
            break;
        }
        images.add(i, d);
        return i;
    }

    std::string complete_file_name(const char *, driver) const;
};

} // namespace detail
} // namespace xpu

#endif
