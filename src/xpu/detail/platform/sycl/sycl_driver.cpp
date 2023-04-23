#include "sycl_driver.h"
#include "../../log.h"
#include "../../config.h"

using namespace xpu::detail;

sycl::queue sycl_driver::default_queue() {
    return m_default_queue;
}

error sycl_driver::setup() {
    if (config::profile) {
        m_prop_list = sycl::property_list{sycl::property::queue::enable_profiling(), sycl::property::queue::in_order()};
    } else {
        m_prop_list = sycl::property_list{sycl::property::queue::in_order()};
    }
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

error sycl_driver::create_queue(void **queue, int device) {
    auto q = std::make_unique<sycl::queue>(sycl::device::get_devices()[device], m_prop_list);
    m_queues.emplace_back(std::move(q));
    *queue = m_queues.back().get();
    return 0;
}

error sycl_driver::destroy_queue(void *queue) {
    auto q = static_cast<sycl::queue *>(queue);
    auto it = std::find_if(m_queues.begin(), m_queues.end(), [q](const auto &q_ptr) { return q_ptr.get() == q; });
    if (it != m_queues.end()) {
        m_queues.erase(it);
    }
    return 0;
}

error sycl_driver::synchronize_queue(void *handle) {
    auto q = get_queue(handle);
    q.wait();
    return 0;
}

error sycl_driver::memcpy(void *dst, const void *src, size_t bytes) {
    m_default_queue.memcpy(dst, src, bytes).wait();
    return 0;
}

error sycl_driver::memcpy_async(void *dst, const void *src, size_t bytes, void *handle, double *ms) {
    auto q = get_queue(handle);
    if (ms == nullptr) {
        q.memcpy(dst, src, bytes);
    } else {
        sycl::event ev = q.memcpy(dst, src, bytes);
        ev.wait();
        double ns = ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                    ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        *ms = ns / 1000000.0;
    }
    return 0;
}

error sycl_driver::memset(void *dst, int ch, size_t bytes) {
    m_default_queue.memset(dst, ch, bytes).wait();
    return 0;
}

error sycl_driver::memset_async(void *dst, int ch, size_t bytes, void *handle, double *ms) {
    auto q = get_queue(handle);
    if (ms == nullptr) {
        q.memset(dst, ch, bytes);
    } else {
        sycl::event ev = q.memset(dst, ch, bytes);
        ev.wait();
        double ns = ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                    ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        *ms = ns / 1000000.0;
    }
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
    props->driver = sycl;

    // XPU_LOG("sycl_driver::get_properties: device name: %s, version: %s", props->name.c_str(), dev.get_info<sycl::info::device::version>().c_str());
    props->arch = dev.get_info<sycl::info::device::version>();

    props->shared_mem_size = dev.get_info<sycl::info::device::local_mem_size>();
    props->const_mem_size = 0; // constant memory deprecated in SYCL 2020

    props->warp_size = dev.get_info<sycl::info::device::sub_group_sizes>()[0];
    props->max_threads_per_block = dev.get_info<sycl::info::device::max_work_group_size>();
    props->max_grid_size = {dev.get_info<sycl::info::device::max_work_item_sizes<3>>()[0],
                            dev.get_info<sycl::info::device::max_work_item_sizes<3>>()[1],
                            dev.get_info<sycl::info::device::max_work_item_sizes<3>>()[2]};

    return 0;
}

error sycl_driver::get_ptr_prop(const void *ptr, int *device, mem_type *type) {
    try {
        sycl::usm::alloc alloc = sycl::get_pointer_type(ptr, m_default_queue.get_context());

        switch (alloc) {
        case sycl::usm::alloc::device:
            *type = mem_device;
            break;
        case sycl::usm::alloc::host:
            *type = mem_host;
            break;
        case sycl::usm::alloc::shared:
            *type = mem_shared;
            break;
        case sycl::usm::alloc::unknown:
            *type = mem_unknown;
            break;
        }

        if (*type == mem_unknown) {
            *device = -1;
            return 0;
        }

        sycl::device dev = sycl::get_pointer_device(ptr, m_default_queue.get_context());
        // XPU_LOG("sycl_driver::pointer_get_device: %s", dev.get_info<sycl::info::device::name>().c_str());
        *device = get_device_id(dev);
    } catch (sycl::exception &e) {
        *type = mem_unknown;
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

const char *sycl_driver::error_to_string(error /*err*/) {
    return "Unknown error";
}

driver_t sycl_driver::get_type() {
    return driver_t::sycl;
}

int sycl_driver::get_device_id(sycl::device dev) {
    std::vector<sycl::device> devices = sycl::device::get_devices();
    auto it = std::find(devices.begin(), devices.end(), dev);
    return std::distance(devices.begin(), it);
}

sycl::queue sycl_driver::get_queue(void *handle) {
    // TODO: one could just use a static_cast here to retrieve the queue, but this is more robust
    auto it = std::find_if(m_queues.begin(), m_queues.end(), [handle](const auto &q_ptr) { return q_ptr.get() == handle; });
    if (it != m_queues.end()) {
        return *it->get();
    }
    throw std::runtime_error("Queue not found");
}

extern "C" xpu::detail::backend_base *create() {
    return new xpu::detail::sycl_driver{};
}

extern "C" void destroy(xpu::detail::backend_base *b) {
    delete b;
}
