#include <xpu/host.h>

#include <cstdio>

int main() {
    xpu::initialize();

    std::vector<xpu::device> devices = xpu::device::all();

    for (auto &d : devices) {
        xpu::device_prop prop{d};

        std::cout << prop.xpuid() << ": " << prop.name();

        if (!prop.arch().empty())
            std::cout << " [" << prop.arch() << "]";

        std::cout << std::endl;
    }

    return 0;
}
