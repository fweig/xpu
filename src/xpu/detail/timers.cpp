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
};

static struct {
    std::vector<timer> stack;
} T;

void xpu::detail::push_timer(std::string_view name) {
    timings t;
    t.name = name;
    t.has_details = config::profile;
    T.stack.push_back(timer{std::chrono::high_resolution_clock::now(), t});
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

void xpu::detail::add_memset_time(double ms) {
    for (auto &t : T.stack) {
        t.ts.memset += ms;
    }
}

void xpu::detail::add_memcpy_time(double ms, direction_t dir) {
    for (auto &t : T.stack) {
        if (dir == dir_h2d) {
            t.ts.copy_h2d += ms;
        } else {
            t.ts.copy_d2h += ms;
        }
    }
}

void xpu::detail::add_kernel_time(const char *name, double ms) {
    for (auto &t : T.stack) {
        auto &k = t.ts.kernels;
        auto it = std::find_if(k.begin(), k.end(),
            [&](const auto &k) { return k.name == name; });
        if (it == k.end()) {
            k.push_back(kernel_timings{name, std::vector<double>{ms}});
        } else {
            it->times.emplace_back(ms);
        }
    }
}
