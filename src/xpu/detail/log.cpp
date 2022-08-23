#include "log.h"

#include <cstdarg>
#include <cstdio>
#include <memory>
#include <string>

using namespace xpu::detail;

logger &logger::instance() {
    static logger the_logger;
    return the_logger;
}

void logger::initialize(std::function<void(const char *)> write_out) {
    this->m_write_out = std::move(write_out);
}

bool logger::active() const {
    return m_write_out != nullptr;
}

void logger::write(const char *formatstr, ...) {
    if (not active()) {
        return;
    }

    std::va_list args;
    va_start(args, formatstr);
    int buf_size = std::vsnprintf(nullptr, 0, formatstr, args);
    va_end(args);

    std::string formatted(size_t(buf_size), '\0');

    va_start(args, formatstr);
    std::vsnprintf(formatted.data(), buf_size + 1, formatstr, args);
    va_end(args);

    m_write_out(formatted.c_str());
}

std::string xpu::detail::format(const char *format, ...) {
    std::va_list args;
    va_start(args, format);
    int buf_size = std::vsnprintf(nullptr, 0, format, args);
    va_end(args);

    std::string formatted(size_t(buf_size), '\0');

    va_start(args, format);
    std::vsnprintf(formatted.data(), buf_size + 1, format, args);
    va_end(args);

    return formatted;
}
