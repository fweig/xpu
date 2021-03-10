#include "TestKernels.h"
#include <xpu/host.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <vector>

TEST(XPUTest, CanRunVectorAdd) {
    constexpr int NElems = 100;

    std::vector<float> hx(NElems, 8);
    std::vector<float> hy(NElems, 8);

    float *dx = xpu::device_malloc<float>(NElems);
    float *dy = xpu::device_malloc<float>(NElems);
    float *dz = xpu::device_malloc<float>(NElems);

    xpu::copy(dx, hx.data(), NElems);
    xpu::copy(dy, hy.data(), NElems);

    xpu::run_kernel<TestKernels::vector_add>(xpu::grid::n_threads(NElems), dx, dy, dz, NElems);

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
    // EXPECT_THROW(xpu::memcpy(nullptr, nullptr, 5), xpu::exception);

    // float *ptr = xpu::device_malloc<float>(10);
    // xpu::free(ptr);
    // EXPECT_THROW(xpu::free(ptr), xpu::exception);
}

TEST(XPUTest, CanSortAny8ByteStruct) {

    constexpr int NElems = 100000;

    std::mt19937 gen{1337};
    std::uniform_int_distribution<unsigned int> dist{0, 1000000};

    std::unordered_set<unsigned int> keys{};
    while (keys.size() < NElems) {
        keys.insert(dist(gen));
    }

    std::vector<key_value_t> items{};
    for (auto key : keys) {
        items.push_back({key, dist(gen)});
    }

    std::vector<key_value_t> itemsSorted = items;
    std::sort(itemsSorted.begin(), itemsSorted.end(), [](key_value_t a, key_value_t b){
        return a.key < b.key;
    });

    // for (auto &x : items) {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;

    key_value_t *ditems = xpu::device_malloc<key_value_t>(NElems);
    key_value_t *buf = xpu::device_malloc<key_value_t>(NElems);
    key_value_t **dst = xpu::device_malloc<key_value_t *>(1);

    xpu::copy(ditems, items.data(), NElems);

    xpu::run_kernel<TestKernels::sort_struct>(xpu::grid::n_blocks(1), ditems, NElems, buf, dst);

    key_value_t *hdst = nullptr;
    xpu::copy(&hdst, dst, 1);
    xpu::copy(items.data(), hdst, NElems);

    // for (size_t i = 1; i < 220; i++) {
    //     printf("gt %lu: %u, %u\n", i, itemsSorted[items.size()-i].key, itemsSorted[items.size()-i].value);
    //     printf("gpu %lu: %u, %u\n", i, items[items.size()-i].key, items[items.size()-i].value);
    // }

    // int start = -1, startZero = -1;
    // for (size_t i = 0; i < items.size(); i++) {
    //     // if (i == 209) {
    //     //     printf("i = 209, cpu key = %u, gpu key = %u\n", itemsSorted[i].key, items[i].key );
    //     // }
    //     if (items[i].key == 0 && startZero == -1) {
    //         startZero = i;
    //     }
    //     if (items[i].key != 0 && startZero != -1) {
    //         printf("Found Zero Seq fromt %u to %lu\n", startZero, i);
    //         startZero = -1;
    //     }
    //     if (items[i].key != itemsSorted[i].key && start == -1) {
    //         start = i;
    //     }
    //     if (items[i].key == itemsSorted[i].key && start != -1) {
    //         printf("%lu: gt %u, gpu %u\n", i-2, itemsSorted[i-2].key, items[i-2].key);
    //         printf("%lu: gt %u, gpu %u\n", i-1, itemsSorted[i-1].key, items[i-1].key);
    //         printf("Seq from %d to %lu is different\n", start, i);
    //         start = -1;
    //     }
    // }


    for (size_t i = 0; i < NElems; i++) {
        EXPECT_EQ(items[i].key, itemsSorted[i].key) << " with i = " << i;
        ASSERT_EQ(items[i].value, itemsSorted[i].value);
    }

    // for (auto &x : items) {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;
}

TEST(XPUTest, CanSortFloatsShort) {

    constexpr int NElems = 100;

    std::mt19937 gen{1337};
    std::uniform_int_distribution<unsigned int> dist{0, 10000};

    std::vector<unsigned int> items{};
    for (size_t i = 0; i < NElems; i++) {
        items.emplace_back(dist(gen));
    }

    std::vector<unsigned int> itemsSorted = items;
    std::sort(itemsSorted.begin(), itemsSorted.end());

    // for (auto &x : items) {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;

    unsigned int *ditems = xpu::device_malloc<unsigned int>(NElems);
    unsigned int *buf = xpu::device_malloc<unsigned int>(NElems);
    unsigned int **dst = xpu::device_malloc<unsigned int *>(1);

    xpu::copy(ditems, items.data(), NElems);

    xpu::run_kernel<TestKernels::sort>(xpu::grid::n_blocks(1), ditems, NElems, buf, dst);

    unsigned int *hdst = nullptr;
    xpu::copy(&hdst, dst, 1);
    xpu::copy(items.data(), hdst, NElems);

    // for (auto &x : items) {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;

    for (size_t i = 0; i < NElems; i++) {
        ASSERT_EQ(items[i], itemsSorted[i]);
    }

}

TEST(XPUTest, CanSetAndReadCMem) {
    test_constants orig{1, 2, 3};
    xpu::hd_buffer<test_constants> out{1};

    xpu::set_cmem<TestKernels>(orig);

    xpu::run_kernel<TestKernels::access_cmem>(xpu::grid::n_threads(1), out.device());
    xpu::copy(out, xpu::device_to_host);

    test_constants result = *out.host();
    EXPECT_EQ(orig.x, result.x);
    EXPECT_EQ(orig.y, result.y);
    EXPECT_EQ(orig.z, result.z);

    // xpu::get_cmem<xpu_test::test_kernels>(constants_copy);

    // ASSERT_EQ(constants_orig.x, constants_copy.x);
    // ASSERT_EQ(constants_orig.y, constants_copy.y);
    // ASSERT_EQ(constants_orig.z, constants_copy.z);
}

TEST(XPUTest, CanGetThreadIdx) {
    xpu::hd_buffer<int> idx{64};

    xpu::run_kernel<TestKernels::get_thread_idx>(xpu::grid::n_threads(64), idx.device());
    xpu::copy(idx, xpu::device_to_host);

    for (int i = 0; i < 64; i++) {
        if (xpu::active_driver() == xpu::driver::cpu) {
            EXPECT_EQ(idx.host()[i], 0) << " with i = " << i;
        } else {
            EXPECT_EQ(idx.host()[i], i);
        }
    }
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
    if (driver_name == "hip") {
        return xpu::driver::hip;
    }

    std::cout << "ERROR: Unknown driver '" << driver_name_c << "'" << std::endl;
    std::abort();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    xpu::initialize(get_target_driver());
    return RUN_ALL_TESTS();
}
