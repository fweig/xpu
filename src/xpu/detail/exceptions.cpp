#include "exceptions.h"

#include "log.h"

#include <stdexcept>

void xpu::detail::throw_out_of_range(std::string_view where, size_t i, size_t size) {
    throw std::out_of_range{format("%.*s: index out of range: i = %zu, size = %zu", int{where.size()}, where.data(), i, size)};
}
