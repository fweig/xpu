#ifndef XPU_DETAIL_LOG_H
#define XPU_DETAIL_LOG_H

#include "macros.h"

#include <functional>
#include <string>
#include <string_view>

namespace xpu::detail {

class logger {

public:
    static logger &instance();

    void initialize(std::function<void(std::string_view)>);
    bool active() const;
    void write(const char *, ...) XPU_ATTR_FORMAT_PRINTF(2, 3);

private:
    std::function<void(std::string_view)> m_write_out;

};

std::string format(const char *, ... ) XPU_ATTR_FORMAT_PRINTF(1, 2);

} // namespace xpu::detail

#define XPU_LOG(format, ...) xpu::detail::logger::instance().write("xpu: " format, ##__VA_ARGS__)

#endif
