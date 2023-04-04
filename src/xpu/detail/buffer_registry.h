#ifndef XPU_DETAIL_BUFFER_REGISTRY_H
#define XPU_DETAIL_BUFFER_REGISTRY_H

#include <atomic>
#include <cstddef>
#include <memory>
#include <unordered_map>

namespace xpu::detail {

enum buffer_type {
    buf_host,
    buf_device,
    buf_shared,
    buf_io,
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

    buffer_data &get(const void *ptr) { return m_entries.at(ptr).data; }

private:
    struct buffer_entry {
        buffer_data data;
        std::unique_ptr<std::atomic<int>> ref_count;
    };

    using buffer_map = std::unordered_map<const void *, buffer_entry>;

    buffer_map m_entries;

    void remove(buffer_map::iterator it);
};

} // namespace xpu::detail

#endif
