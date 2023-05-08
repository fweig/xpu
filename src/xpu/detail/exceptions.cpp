#include "exceptions.h"

#include "log.h"

#include <stdexcept>

void xpu::detail::throw_out_of_range(std::string_view where, size_t i, size_t size) {
    throw std::out_of_range{format("%.*s: index out of range: i = %zu, size = %zu",
        static_cast<int>(where.size()), where.data(), i, size)};
}

void xpu::detail::throw_invalid_argument(std::string_view where, std::string_view what) {
    throw std::invalid_argument{format("%.*s: invalid argument: %.*s",
        static_cast<int>(where.size()), where.data(),
        static_cast<int>(what.size()), what.data())};
}
