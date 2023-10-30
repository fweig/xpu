#include <xpu/host.h>

#include <cstdio>

int main(int argc, char **argv) {
    xpu::settings settings;
    xpu::initialize(settings);

    bool verbose = false;

    if (argc > 1 && std::string(argv[1]) == "-v")
        verbose = true;

    std::vector<xpu::device> devices = xpu::device::all();

    for (auto &d : devices) {
        xpu::device_prop prop{d};

        if (!verbose) {
            std::cout << prop.xpuid() << ": " << prop.name();

            if (!prop.arch().empty())
                std::cout << " [" << prop.arch() << "]";

            std::cout << std::endl;

            continue;
        }

        std::cout << "Device: " << prop.xpuid() << std::endl;
        std::cout << "  Name: " << prop.name() << std::endl;
        std::cout << "  Driver: " << prop.backend() << std::endl;
        std::cout << "  Architecture: " << prop.arch() << std::endl;
        std::cout << "  Shared memory size: " << prop.shared_mem_size() << std::endl;
        std::cout << "  Constant memory size: " << prop.const_mem_size() << std::endl;
        std::cout << "  Warp size: " << prop.warp_size() << std::endl;
        std::cout << "  Max threads per block: " << prop.max_threads_per_block() << std::endl;
        std::cout << "  Max grid size: " << prop.max_grid_size()[0] << "x" << prop.max_grid_size()[1] << "x" << prop.max_grid_size()[2] << std::endl;
        std::cout << "  Global memory total: " << prop.global_mem_total() << std::endl;
        std::cout << "  Global memory available: " << prop.global_mem_available() << std::endl;
    }

    return 0;
}
