#ifndef XPU_DETAIL_LOG_H
#define XPU_DETAIL_LOG_H

#include <functional>

namespace xpu {
namespace detail {

class logger {

public:
    static logger &instance();

    void initialize(std::function<void(const char *)>);
    bool active() const;
    void write(const char *, ...) __attribute__ ((format (__printf__, 2, 3)));

private:
    std::function<void(const char *)> write_out;

};

} // namespace detail
} // namespace xpu

#define XPU_LOG(format, ...) xpu::detail::logger::instance().write("xpu: " format, ##__VA_ARGS__)

#endif
