#include "buffer_registry.h"
#include "../host.h"

using namespace xpu::detail;

buffer_registry &buffer_registry::instance() {
    static buffer_registry instance;
    return instance;
}

void *buffer_registry::create(size_t size, buffer_type type, void *host_ptr) {
    void *ptr = nullptr;
    bool owns_host_ptr = false;
    switch (type) {
    case buf_host:
        ptr = xpu::malloc_host(size);
        if (host_ptr != nullptr) {
            std::memcpy(ptr, host_ptr, size);
        }
        host_ptr = ptr;
        break;
    case buf_device:
        ptr = xpu::malloc_device(size);
        host_ptr = nullptr;
        break;
    case buf_shared:
        ptr = xpu::malloc_shared(size);
        if (host_ptr != nullptr) {
            std::memcpy(ptr, host_ptr, size);
        }
        host_ptr = ptr;
        break;
    case buf_io: {
            if (host_ptr == nullptr) {
                host_ptr = xpu::malloc_host(size);
                owns_host_ptr = true;
            }
            xpu::device active_dev = xpu::device::active();
            if (active_dev.backend() == xpu::cpu) {
                ptr = host_ptr;
            } else {
                ptr = xpu::malloc_device(size);
            }
            break;
        }
    }

    buffer_data data {
        ptr,
        host_ptr,
        owns_host_ptr,
        type,
        size
    };

    m_entries.emplace(ptr, buffer_entry{
        data,
        std::make_unique<std::atomic<int>>(1)
    });
    XPU_LOG("Created buffer: %p", ptr);
    return ptr;
}

void buffer_registry::add_ref(const void *ptr) {
    auto &entry = m_entries.at(ptr);
    XPU_LOG("Add ref: %p", ptr);
    (*entry.ref_count)++;
}

void buffer_registry::remove_ref(const void *ptr) {
    auto it = m_entries.find(ptr);
    if (it == m_entries.end()) {
        return;
    }
    XPU_LOG("Removing ref: %p", ptr);
    (*it->second.ref_count)--;
    if (*it->second.ref_count == 0) {
        XPU_LOG("Free buffer: %p", ptr);
        remove(it);
    }
}

void buffer_registry::remove(buffer_map::iterator it) {
    if (it == m_entries.end()) {
        return;
    }
    auto &entry = it->second;
    switch (entry.data.type) {
    case buf_host:
    case buf_device:
    case buf_shared:
        xpu::free(entry.data.ptr);
        break;
    case buf_io: {
            xpu::device active_dev = xpu::device::active();
            if (active_dev.backend() != xpu::cpu) {
                xpu::free(entry.data.ptr);
            }
            if (entry.data.owns_host_ptr) {
                xpu::free(entry.data.host_ptr);
            }
            break;
        }
    }
    m_entries.erase(it);
}
