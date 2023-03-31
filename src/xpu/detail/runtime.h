#ifndef XPU_DETAIL_RUNTIME_H
#define XPU_DETAIL_RUNTIME_H

#include "../common.h"
#include "../driver/cpu/cpu_driver.h"
#include "common.h"
#include "dl_utils.h"
#include "dynamic_loader.h"
#include "log.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xpu {
class ptr_prop;
struct settings;
}

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

    void initialize(const settings &);

    void *malloc_host(size_t);
    void *malloc_device(size_t);
    void *malloc_shared(size_t);

    void free(void *);
    void memcpy(void *, const void *, size_t);
    void memset(void *, int, size_t);

    std::vector<detail::device> get_devices() { return m_devices; }
    detail::device active_device() const { return m_active_device; }
    detail::device get_device(int id) const { return m_devices.at(id); }
    detail::device get_device(driver_t driver, int id) const;
    detail::device get_device(std::string_view name) const;
    detail::device_prop device_properties(int id);

    int device_get_id(driver_t backend, int device_nr);
    void get_ptr_prop(const void *, ptr_prop *);

    template<typename Kernel, typename... Args>
    void run_kernel(grid g, Args&&... args) {
        static_assert(std::is_same_v<typename Kernel::tag, kernel_tag>);

        float ms;
        error err = get_image<Kernel>()->template run_kernel<Kernel>((m_measure_time ? &ms : nullptr), get_active_driver(), g, std::forward<Args>(args)...);
        throw_on_driver_error(m_active_device.backend, err);

        if (m_measure_time) {
            size_t id = linear_type_id<Kernel>::get();

            if (m_profiling.size() <= id) {
                m_profiling.resize(id+1);
            }

            m_profiling.at(id).push_back(ms);
        }
    }

    template<typename Func, typename... Args>
    void call(Args&&... args) {
        static_assert(std::is_same_v<typename Func::tag, function_tag>);
        error err = get_image<Func>()->template call<Func>(std::forward<Args>(args)...);
        throw_on_driver_error(m_active_device.backend, err);
    }

    template<typename C>
    void set_constant(const typename C::data_t &symbol) {
        static_assert(std::is_same_v<typename C::tag, constant_tag>);
        error err = get_image<C>()->template set<C>(symbol);
        throw_on_driver_error(m_active_device.backend, err);
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
    std::unique_ptr<lib_obj<driver_interface>> m_sycl_driver;

    image_pool m_images;

    bool m_measure_time = false;
    std::vector<std::vector<float>> m_profiling;

    detail::device m_active_device;
    std::vector<detail::device> m_devices;

    static bool getenv_bool(std::string name, bool fallback);
    static std::string getenv_str(std::string name, std::string_view fallback);

    template<typename A>
    image<typename A::image> *get_image() {
        auto *img = m_images.find< image<typename A::image> >(m_active_device.backend);
        if (img == nullptr) {
            img = load_image<typename A::image>(m_active_device.backend);
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
        case sycl:
            i = new image<I>(complete_file_name(image_file_name<I>{}(), d).c_str());
            break;
        }
        m_images.add(i, d);
        return i;
    }

    std::optional<detail::device> try_parse_device(std::string_view) const;

    bool has_driver(driver_t) const;
    driver_interface *get_driver(driver_t) const;
    driver_interface *get_active_driver() const;

    std::string complete_file_name(const char *, driver_t) const;

    const char *driver_str(driver_t, bool lower = false) const;

    void throw_on_driver_error(driver_t, error) const;

    void raise_error_if(bool, std::string_view) const;

    // Exception are raised from a seperate function, to avoid circular dependency with host.h header/
    [[noreturn]] void raise_error(std::string_view) const;

};

} // namespace xpu::detail

#endif
