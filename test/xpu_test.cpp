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

    xpu::run_kernel<vector_add>(xpu::grid::n_threads(NElems), dx, dy, dz, NElems);

    std::vector<float> hz(NElems);
    xpu::copy(hz.data(), dz, NElems);

    for (auto &x: hz) {
        ASSERT_EQ(16, x);
    }

    xpu::free(dx);
    xpu::free(dy);
    xpu::free(dz);
}

TEST(XPUTest, CanSortStruct) {

    constexpr size_t NElems = 1000000000;

    std::mt19937 gen{1337};
    std::uniform_int_distribution<unsigned int> dist{0, 1000000};

    key_value_t *ditems = xpu::device_malloc<key_value_t>(NElems);
    key_value_t *buf = xpu::device_malloc<key_value_t>(NElems);
    key_value_t **dst = xpu::device_malloc<key_value_t *>(1);


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


    xpu::copy(ditems, items.data(), NElems);

    xpu::run_kernel<sort_struct>(xpu::grid::n_blocks(1), ditems, NElems, buf, dst);

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

    constexpr int NElems = 128;

    std::mt19937 gen{1337};
    std::uniform_real_distribution<float> dist{0, 10000};

    std::vector<float> items{};
    for (size_t i = 0; i < NElems; i++) {
        items.emplace_back(dist(gen));
    }

    std::vector<float> itemsSorted = items;
    std::sort(itemsSorted.begin(), itemsSorted.end());

    // for (auto &x : items) {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;

    float *ditems = xpu::device_malloc<float>(NElems);
    float *buf = xpu::device_malloc<float>(NElems);
    float **dst = xpu::device_malloc<float *>(1);

    xpu::copy(ditems, items.data(), NElems);

    xpu::run_kernel<sort_float>(xpu::grid::n_blocks(1), ditems, NElems, buf, dst);

    float *hdst = nullptr;
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

template<typename K>
void testMergeKernel(size_t M, size_t N) {
    xpu::hd_buffer<float> a{M};
    xpu::hd_buffer<float> b{N};
    xpu::hd_buffer<float> dst{a.size() + b.size()};

    std::mt19937 gen{1337};
    std::uniform_real_distribution<float> dist{0, 100000};

    std::sort(a.host(), a.host() + a.size());
    std::sort(b.host(), b.host() + b.size());

    xpu::copy(a, xpu::host_to_device);
    xpu::copy(b, xpu::host_to_device);

    xpu::run_kernel<K>(xpu::grid::n_blocks(1), a.device(), a.size(), b.device(), b.size(), dst.device());

    xpu::copy(dst, xpu::device_to_host);

    float *h = dst.host();
    bool isSorted = true;
    for (size_t i = 1; i < dst.size(); i++) {
        isSorted &= (h[i-1] <= h[i]);
    }

    if (!isSorted) {
        for (size_t i = 0; i < dst.size(); i++) {
            std::cout << h[i] << " ";
            if (i % 10 == 9) {
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    ASSERT_TRUE(isSorted);
}

TEST(XPUTest, CanMergeEvenNumberOfItems) {
    testMergeKernel<merge>(512, 512);
}

TEST(XPUTest, CanMergeUnevenNumberOfItems) {
    testMergeKernel<merge>(610, 509);
}

TEST(XPUTest, CanMergeWithOneItemPerThread) {
    testMergeKernel<merge_single>(512, 512);
}

TEST(XPUTest, CanSetAndReadCMem) {
    float3_ orig{1, 2, 3};
    xpu::hd_buffer<float3_> out{1};

    xpu::set_constant<test_constants>(orig);

    xpu::run_kernel<access_cmem>(xpu::grid::n_threads(1), out.device());
    xpu::copy(out, xpu::device_to_host);

    float3_ result = *out.host();
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

    xpu::run_kernel<get_thread_idx>(xpu::grid::n_threads(64), idx.device());
    xpu::copy(idx, xpu::device_to_host);

    for (int i = 0; i < 64; i++) {
        if (xpu::active_driver() == xpu::driver::cpu) {
            EXPECT_EQ(idx.host()[i], 0) << " with i = " << i;
        } else {
            EXPECT_EQ(idx.host()[i], i);
        }
    }
}

TEST(XPUTest, CollectsTimingData) {
    constexpr int NElems = 100000;

    xpu::hd_buffer<float> a{NElems};
    xpu::hd_buffer<float> b{NElems};
    xpu::hd_buffer<float> c{NElems};

    std::fill_n(a.host(), a.size(), 24);
    std::fill_n(b.host(), b.size(), 24);

    xpu::copy(a, xpu::host_to_device);
    xpu::copy(b, xpu::host_to_device);

    for (int i = 0; i < 10; i++) {
        xpu::run_kernel<vector_add_timing>(xpu::grid::n_threads(NElems), a.device(), b.device(), c.device(), NElems);
    }

    auto timings = xpu::get_timing<vector_add_timing>();

    ASSERT_EQ(timings.size(), 10);

    for (auto &t : timings) {
        ASSERT_GT(t, 0.f);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    setenv("XPU_PROFILE", "1", 1); // always enable profiling in unittests
    xpu::initialize(xpu::driver::cpu);
    return RUN_ALL_TESTS();
}
