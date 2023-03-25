#include "sycl_driver.h"
#include "../../detail/log.h"


using namespace xpu::detail;

sycl::queue &sycl_driver::default_queue() {
    return m_default_queue;
}

error sycl_driver::setup() {
    // TODO: check if profiling was enabled by user
    m_prop_list = sycl::property_list{sycl::property::queue::enable_profiling()};
    m_default_queue = sycl::queue(sycl::default_selector_v, m_prop_list);
    return 0;
}

error sycl_driver::malloc_device(void **ptr, size_t bytes) {
    *ptr = sycl::malloc_device(bytes, m_default_queue);
    return 0;
}

error sycl_driver::malloc_host(void **ptr, size_t bytes) {
    *ptr = sycl::malloc_host(bytes, m_default_queue);
    return 0;
}

error sycl_driver::malloc_shared(void **ptr, size_t bytes) {
    *ptr = sycl::malloc_shared(bytes, m_default_queue);
    return 0;
}

error sycl_driver::free(void *ptr) {
    sycl::free(ptr, m_default_queue);
    return 0;
}

error sycl_driver::memcpy(void *dst, const void *src, size_t bytes) {
    m_default_queue.memcpy(dst, src, bytes).wait();
    return 0;
}

error sycl_driver::memset(void *dst, int ch, size_t bytes) {
    m_default_queue.memset(dst, ch, bytes).wait();
    return 0;
}

error sycl_driver::num_devices(int *devices) {
    *devices = sycl::device::get_devices().size();
    return 0;
}

error sycl_driver::set_device(int device) {
    m_default_queue = sycl::queue(sycl::device::get_devices()[device], m_prop_list);
    m_device = device;
    return 0;
}

error sycl_driver::get_device(int *device) {
    *device = m_device;
    return 0;
}

error sycl_driver::device_synchronize() {
    m_default_queue.wait();
    return 0;
}

error sycl_driver::get_properties(device_prop *props, int device) {
    sycl::device dev = sycl::device::get_devices().at(device);
    props->name = dev.get_info<sycl::info::device::name>();
    props->driver = xpu::sycl;

    // XPU_LOG("sycl_driver::get_properties: device name: %s, version: %s", props->name.c_str(), dev.get_info<sycl::info::device::version>().c_str());
    props->major = 0; // dev.get_info<sycl::info::device::version>();
    props->minor = 0;
    return 0;
}

error sycl_driver::pointer_get_device(const void *ptr, int *device) {
    try {
        sycl::device dev = sycl::get_pointer_device(ptr, m_default_queue.get_context());
        // XPU_LOG("sycl_driver::pointer_get_device: %s", dev.get_info<sycl::info::device::name>().c_str());
        *device = get_device_id(dev);
    } catch (sycl::exception &e) {
        // XPU_LOG("sycl_driver::pointer_get_device: %s", e.what());
        *device = -1;
    }
    return 0;
}

error sycl_driver::meminfo(size_t *free, size_t *total) {
    sycl::device device = m_default_queue.get_device();
    *free = device.get_info<sycl::info::device::global_mem_size>(); // no way to get available memory afaik yet
    *total = device.get_info<sycl::info::device::global_mem_size>();
    return 0;
}

const char *sycl_driver::error_to_string(error err) {
    return "Unknown error";
}

xpu::driver_t sycl_driver::get_type() {
    return driver_t::sycl;
}

int sycl_driver::get_device_id(sycl::device dev) {
    std::vector<sycl::device> devices = sycl::device::get_devices();
    auto it = std::find(devices.begin(), devices.end(), dev);
    return std::distance(devices.begin(), it);
}

extern "C" xpu::detail::driver_interface *create() {
    return new xpu::detail::sycl_driver{};
}

extern "C" void destroy(xpu::detail::driver_interface *b) {
    delete b;
}
