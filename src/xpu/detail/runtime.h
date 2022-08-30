#ifndef XPU_DETAIL_RUNTIME_H
#define XPU_DETAIL_RUNTIME_H

#include "../common.h"
#include "../driver/cpu/cpu_driver.h"
#include "dl_utils.h"
#include "dynamic_loader.h"
#include "log.h"

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace xpu::detail {

class image_pool {

public:
    template<typename I>
    I *find(driver_t driver) {
        for (auto &e : entries) {
            if (e.driver == driver && e.id == grouped_type_id<I, image_pool>::get()) {
                return static_cast<I *>(e.image);
            }
        }
        return nullptr;
    }

    template<typename I>
    void add(I *i, driver_t driver) {
        entries.push_back(basic_entry{driver, grouped_type_id<I, image_pool>::get(), i});
    }

private:
    struct basic_entry {
        driver_t driver;
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

    // FIXME this clashes / is ambigious with private function get_active_driver
    driver_t active_driver() const { return m_active_driver; }

    device_prop pointer_get_device(const void *);

    template<typename Kernel, typename... Args>
    void run_kernel(grid g, Args&&... args) {
        float ms;
        error err = get_image<Kernel>()->template run_kernel<Kernel>((m_measure_time ? &ms : nullptr), g, std::forward<Args>(args)...);
        throw_on_driver_error(active_driver(), err);

        if (m_measure_time) {
            size_t id = linear_type_id<Kernel>::get();

            if (m_profiling.size() <= id) {
                m_profiling.resize(id+1);
            }

            m_profiling.at(id).push_back(ms);
        }
    }

    template<typename C>
    void set_constant(const typename C::data_t &symbol) {
        error err = get_image<C>()->template set<C>(symbol);
        throw_on_driver_error(active_driver(), err);
    }

    template<typename Kernel>
    std::vector<float> get_timing() {
        size_t id = linear_type_id<Kernel>::get();

        // Profiling not enabled or kernel hasn't run yet.
        if (m_profiling.size() <= id) {
            return {};
        }

        return m_profiling.at(id);
    }

private:
    std::unique_ptr<cpu_driver> m_cpu_driver;
    std::unique_ptr<lib_obj<driver_interface>> m_cuda_driver;
    std::unique_ptr<lib_obj<driver_interface>> m_hip_driver;

    driver_t m_active_driver = cpu;

    image_pool m_images;

    bool m_measure_time = false;
    std::vector<std::vector<float>> m_profiling;

    std::vector<device_prop> m_devices;
    std::array<std::vector<device_prop>, 3> m_devices_by_driver;

    static bool getenv_bool(std::string name, bool fallback);
    static std::string getenv_str(std::string name, std::string_view fallback);

    template<typename A>
    image<typename A::image> *get_image() {
        auto *img = m_images.find< image<typename A::image> >(active_driver());
        if (img == nullptr) {
            img = load_image<typename A::image>(active_driver());
        }
        if (img == nullptr) {
            raise_error(format("Failed to load image for kernel '%s'", type_name<A>()));
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
            i = new image<I>(complete_file_name(image_file_name<I>{}(), d).c_str());
            break;
        }
        m_images.add(i, d);
        return i;
    }

    bool has_driver(driver_t) const;
    driver_interface *get_driver(driver_t) const;
    driver_interface *get_active_driver() const;

    std::string complete_file_name(const char *, driver_t) const;

    const char *driver_str(driver_t) const;

    void throw_on_driver_error(driver_t, error) const;

    // Exception are raised from a seperate function, to avoid circular dependency with host.h header/
    [[noreturn]] void raise_error(std::string_view) const;

};

} // namespace xpu::detail

#endif
