#include "test_kernels.h"
#include <xpu/host.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <random>

TEST(XPUTest, CanRunVectorAdd) {
    constexpr int NElems = 100;

    std::vector<float> hx(NElems, 8);
    std::vector<float> hy(NElems, 8);

    float *dx = xpu::device_malloc<float>(NElems);
    float *dy = xpu::device_malloc<float>(NElems);
    float *dz = xpu::device_malloc<float>(NElems);

    xpu::copy(dx, hx.data(), NElems);
    xpu::copy(dy, hy.data(), NElems);

    xpu::run_kernel<xpu_test::vector_add>(xpu::grid::n_threads(NElems), dx, dy, dz, NElems);

    std::vector<float> hz(NElems);
    xpu::copy(hz.data(), dz, NElems);

    for (auto &x: hz) {
        ASSERT_EQ(16, x);
    }

    xpu::free(dx);
    xpu::free(dy);
    xpu::free(dz);
}

TEST(XPUTest, ThrowsExceptionsOnError) {
    EXPECT_THROW(xpu::memcpy(nullptr, nullptr, 5), xpu::exception);

    float *ptr = xpu::device_malloc<float>(10);
    xpu::free(ptr);
    EXPECT_THROW(xpu::free(ptr), xpu::exception);
}

TEST(XPUTest, CanSortFloats) {
    constexpr int NElems = 10;

    std::mt19937 gen{1337};
    std::uniform_real_distribution<float> dist{};

    std::vector<float> items{};
    for (size_t i = 0; i < NElems; i++) {
        items.emplace_back(dist(gen));
    }

    // for (auto &x : items) {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;

    float *ditems = xpu::device_malloc<float>(NElems);

    xpu::copy(ditems, items.data(), NElems);

    xpu::run_kernel<xpu_test::sort_floats>(xpu::grid::n_threads(1), ditems, NElems);

    xpu::copy(items.data(), ditems, NElems);

    for (size_t i = 1; i < NElems; i++) {
        EXPECT_LE(items[i-1], items[i]);
    }

    // for (auto &x : items) {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;
}

xpu::driver get_target_driver() {
    const char *driver_name_c = std::getenv("XPU_TEST_DRIVER");
    if (driver_name_c == nullptr) {
        return xpu::driver::cpu;
    }
    std::string driver_name{driver_name_c};
    std::transform(driver_name.begin(), driver_name.end(), driver_name.begin(), [](unsigned char c) { return std::tolower(c); });

    if (driver_name == "cpu") {
        return xpu::driver::cpu;
    }
    if (driver_name == "cuda") {
        return xpu::driver::cuda;
    }

    std::cout << "ERROR: Unknown driver '" << driver_name_c << "'" << std::endl;
    std::abort();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    xpu::initialize(get_target_driver());
    return RUN_ALL_TESTS();
}