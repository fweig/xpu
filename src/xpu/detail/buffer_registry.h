#ifndef XPU_DETAIL_BUFFER_REGISTRY_H
#define XPU_DETAIL_BUFFER_REGISTRY_H

#include "common.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

namespace xpu::detail {

enum buffer_type {
    buf_pinned,
    buf_device,
    buf_managed,
    buf_io,
    buf_stack,
};

struct buffer_data {
    void *ptr;
    void *host_ptr;
    bool owns_host_ptr;
    buffer_type type;
    size_t size;
};

class buffer_registry {
public:
    static buffer_registry &instance();

    void *create(size_t size, buffer_type type, void *host_ptr = nullptr);
    void add_ref(const void *ptr);
    void remove_ref(const void *ptr);
    buffer_data get(const void *ptr);

    void stack_alloc(device dev, size_t size);
    void *stack_push(device dev, size_t size);
    void stack_pop(device dev, void *ptr);
    bool stack_contains(const void *ptr);

private:
    struct buffer_entry {
        buffer_data data;
        std::unique_ptr<std::atomic<int>> ref_count;
    };
    using buffer_map = std::unordered_map<const void *, buffer_entry>;
    buffer_map m_entries;

    static constexpr size_t stack_alignment = 256;
    struct stack_entry {
        void *start;
        size_t size;
        std::vector<std::pair<void *, size_t>> alloced_blocks; // head, size
        ~stack_entry();
        void *head() const;
        bool contains(const void *ptr) const;
    };
    std::unordered_map<int, std::unique_ptr<stack_entry>> m_stacks; // device id -> stack

    void remove(buffer_map::iterator it);
};

} // namespace xpu::detail

#endif
