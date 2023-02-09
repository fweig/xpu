#include <xpu/host.h>

#include <cstdio>

int main() {
    xpu::initialize();

    std::vector<xpu::device_prop> devices = xpu::get_devices();

    for (auto &d : devices) {
        printf("%s: %s [%d%d]\n", d.xpuid.c_str(), d.name.c_str(), d.major, d.minor);
    }

    return 0;
}
