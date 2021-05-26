#include "log.h"

#include <cstdarg>
#include <cstdio>
#include <memory>

using namespace xpu::detail;

logger &logger::instance() {
    static logger the_logger;
    return the_logger;
}

void logger::initialize(std::function<void(const char *)> write_out) {
    this->write_out = std::move(write_out);
}

void logger::write(const char *format, ...) {
    if (write_out == nullptr) {
        return;
    }

    std::va_list args;
    va_start(args, format);
    int buf_size = std::vsnprintf(nullptr, 0, format, args) + 1;
    va_end(args);

    std::unique_ptr<char[]> formatted{new char[buf_size]};

    va_start(args, format);
    std::vsnprintf(formatted.get(), buf_size, format, args);
    va_end(args);

    write_out(formatted.get());
}
