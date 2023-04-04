#include "buffer_registry.h"
#include "runtime.h"
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
    case buf_stack:
        ptr = stack_push(runtime::instance().active_device(), size);
        break;
    }

    if (type != buf_stack) {
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
    }
    XPU_LOG("Created buffer: %p", ptr);
    return ptr;
}

void buffer_registry::add_ref(const void *ptr) {
    auto it = m_entries.find(ptr);
    if (it == m_entries.end()) {
        return;
    }
    XPU_LOG("Add ref: %p", it->second.data.ptr);
    (*it->second.ref_count)++;
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

buffer_data buffer_registry::get(const void *ptr) {
    auto it = m_entries.find(ptr);
    if (it != m_entries.end()) {
        return it->second.data;
    }
    // Check if the pointer is in a stack
    if (!stack_contains(ptr)) {
        throw std::runtime_error("Buffer not found");
    }

    for (auto &it : m_stacks) {
        for (auto &block : it.second->alloced_blocks) {
            if (block.first == ptr) {
                return buffer_data{
                    const_cast<void *>(ptr),
                    nullptr,
                    false,
                    buf_stack,
                    block.second
                };
            }
        }
    }

    throw std::runtime_error("Internal error: Couldn't find stack buffer. This should never happen.");
}

void buffer_registry::stack_alloc(device dev, size_t size) {
    auto it = m_stacks.find(dev.id);
    if (it != m_stacks.end()) {
        throw std::runtime_error("Stack already allocated");
    }
    void *ptr = xpu::malloc_device(size);
    m_stacks.emplace(dev.id, nullptr);
    m_stacks[dev.id].reset(new stack_entry{
        ptr,
        size,
        std::vector<std::pair<void *, size_t>>{}
    });
}

void *buffer_registry::stack_push(device dev, size_t size) {
    auto it = m_stacks.find(dev.id);
    if (it == m_stacks.end()) {
        throw std::runtime_error("Stack not allocated");
    }
    auto &stack = it->second;
    void *head = stack->head();
    // align to 256 bytes
    head = static_cast<char *>(head) + (stack_alignment - (reinterpret_cast<size_t>(head) % stack_alignment));

    if (static_cast<char *>(head) + size > static_cast<char *>(stack->start) + stack->size) {
        throw std::runtime_error("Stack overflow");
    }

    stack->alloced_blocks.emplace_back(head, size);

    return head;
}

void buffer_registry::stack_pop(device dev, void *ptr) {
    auto it = m_stacks.find(dev.id);
    if (it == m_stacks.end()) {
        throw std::runtime_error("Stack not allocated");
    }
    auto &stack = it->second;

    if (ptr == nullptr) {
        stack->alloced_blocks.clear();
        return;
    }

    for (auto it = stack->alloced_blocks.begin(); it != stack->alloced_blocks.end(); it++) {
        if (it->first == ptr) {
            // Erase this block and all blocks after it
            stack->alloced_blocks.erase(it, stack->alloced_blocks.end());
            return;
        }
    }

    throw std::runtime_error("Stack entry not found");
}

bool buffer_registry::stack_contains(const void *ptr) {
    if (ptr == nullptr) {
        return false;
    }

    for (auto &it : m_stacks) {
        auto &stack = it.second;
        if (stack->contains(ptr)) {
            return true;
        }
    }

    return false;
}

buffer_registry::stack_entry::~stack_entry() {
    printf("Freeing stack: %p\n", start);
    xpu::free(start);
}

void *buffer_registry::stack_entry::head() const {
    if (alloced_blocks.empty()) {
        return start;
    } else {
        return static_cast<char *>(alloced_blocks.back().first) + alloced_blocks.back().second;
    }
}

bool buffer_registry::stack_entry::contains(const void *ptr) const {
    return ptr >= start && ptr < head();
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
    case buf_stack:
        // stack buffer shouldn't be added to m_entries...
        throw std::runtime_error("Internal error: Tried to free a stack buffer. This should never happen.");
    }
    m_entries.erase(it);
}
