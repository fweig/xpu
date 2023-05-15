#include "timers.h"
#include "config.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

using namespace xpu::detail;

using MS = std::chrono::duration<double, std::milli>;

struct timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    timings ts;

    timer() : start(std::chrono::high_resolution_clock::now()) {}
};

static struct {
    std::vector<timer> stack;
} T;

void xpu::detail::push_timer(std::string_view name) {
    timer &t = T.stack.emplace_back();
    t.ts.name = name;
    t.ts.has_details = config::profile;
}

timings xpu::detail::pop_timer() {
    if (T.stack.empty()) {
        throw std::runtime_error("xpu::pop_timer call, but timer stack is empty");
    }

    auto t = T.stack.back();
    T.stack.pop_back();

    auto end = std::chrono::high_resolution_clock::now();
    MS duration = std::chrono::duration_cast<MS>(end - t.start);
    t.ts.wall = duration.count();

    if (!T.stack.empty()) {
        T.stack.back().ts.children.push_back(t.ts);
    }

    return t.ts;
}

void xpu::detail::add_memset_time(double ms, size_t bytes) {
    for (auto &t : T.stack) {
        t.ts.memset += ms;
        t.ts.bytes_memset += bytes;
    }
}

void xpu::detail::add_memcpy_time(double ms, direction_t dir, size_t bytes) {
    for (auto &t : T.stack) {
        if (dir == dir_h2d) {
            t.ts.copy_h2d += ms;
            t.ts.bytes_h2d += bytes;
        } else {
            t.ts.copy_d2h += ms;
            t.ts.bytes_d2h += bytes;
        }
    }
}

void xpu::detail::add_kernel_time(std::string_view name, double ms) {
    for (auto &t : T.stack) {
        auto &k = t.ts.kernels;
        auto it = std::find_if(k.begin(), k.end(),
            [&](const auto &krnl) { return krnl.name == name; });
        if (it == k.end()) {
            k.emplace_back(name, ms);
        } else {
            it->times.emplace_back(ms);
        }
    }
}

void xpu::detail::add_bytes_timer(size_t bytes) {
    T.stack.back().ts.bytes_input += bytes;
}

void xpu::detail::add_bytes_kernel(std::string_view name, size_t bytes) {

    for (auto &t : T.stack) {
        auto &k = t.ts.kernels;
        auto it = std::find_if(k.begin(), k.end(),
            [&](const auto &krnl) { return krnl.name == name; });

        if (it == k.end()) {
            k.emplace_back(name);
            it = k.end() - 1;
        }
        it->bytes_input += bytes;
    }

}
