#include "TestKernels.h"
#include <xpu/host.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <vector>

TEST(XPUTest, CanCreatePointerBuffer) {
    // Test for regression with ambigious free
    // This only has to compile
    xpu::d_buffer<key_value_t *> buf1{100};
    xpu::hd_buffer<key_value_t *> buf2{100};
}

TEST(XPUTest, CanGetDeviceFromPointer) {
    xpu::device_prop active_device = xpu::device_properties();

    int h = 0;
    xpu::device_prop prop = xpu::pointer_get_device(&h);
    ASSERT_EQ(prop.driver, xpu::cpu);

    xpu::hd_buffer<int> hdbuf{1};
    prop = xpu::pointer_get_device(hdbuf.h());
    ASSERT_EQ(prop.driver, xpu::cpu);
    prop = xpu::pointer_get_device(hdbuf.d());
    ASSERT_EQ(prop.name, active_device.name);

    xpu::d_buffer<int> dbuf{1};
    prop = xpu::pointer_get_device(dbuf.d());
    ASSERT_EQ(prop.name, active_device.name);
}

TEST(XPUTest, CanConvertTypenamesToString) {
    ASSERT_STREQ(xpu::detail::type_name<int>(), "int");
    ASSERT_STREQ(xpu::detail::type_name<xpu::device_prop>(), "xpu::device_prop");
}

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

    constexpr size_t NElems = 1000000;

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

    for (size_t i = 0; i < M; i++) {
        a.h()[i] = dist(gen);
    }
    std::sort(a.h(), a.h() + a.size());

    for (size_t i = 0; i < N; i++) {
        b.h()[i] = dist(gen);
    }
    std::sort(b.h(), b.h() + b.size());

    xpu::copy(a, xpu::host_to_device);
    xpu::copy(b, xpu::host_to_device);

    xpu::run_kernel<K>(xpu::grid::n_blocks(1), a.d(), a.size(), b.d(), b.size(), dst.d());

    xpu::copy(dst, xpu::device_to_host);

    float *h = dst.h();
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

TEST(XPUTest, CanRunBlockScan) {
    size_t blockSize = xpu::active_driver() == xpu::cpu ? 1 : 64;

    xpu::hd_buffer<int> incl{blockSize};
    xpu::hd_buffer<int> excl{blockSize};

    xpu::run_kernel<block_scan>(xpu::grid::n_blocks(1), incl.d(), excl.d());

    xpu::copy(incl, xpu::device_to_host);
    xpu::copy(excl, xpu::device_to_host);

    int inclSum = 1;
    int exclSum = 0;

    for (size_t i = 0; i < blockSize; i++) {
        ASSERT_EQ(incl.h()[i], inclSum);
        ASSERT_EQ(excl.h()[i], exclSum);
        inclSum++;
        exclSum++;
    }
}

TEST(XPUTest, CanSetAndReadCMem) {
    float3_ orig{1, 2, 3};
    xpu::hd_buffer<float3_> out{1};

    xpu::set_constant<test_constants>(orig);

    xpu::run_kernel<access_cmem>(xpu::grid::n_threads(1), out.d());
    xpu::copy(out, xpu::device_to_host);

    float3_ result = *out.h();
    EXPECT_EQ(orig.x, result.x);
    EXPECT_EQ(orig.y, result.y);
    EXPECT_EQ(orig.z, result.z);

    // xpu::get_cmem<xpu_test::test_kernels>(constants_copy);

    // ASSERT_EQ(constants_orig.x, constants_copy.x);
    // ASSERT_EQ(constants_orig.y, constants_copy.y);
    // ASSERT_EQ(constants_orig.z, constants_copy.z);
}

void test_thread_position(xpu::dim gpu_block_size, xpu::dim gpu_grid_dim) {

    xpu::dim nthreads{
        gpu_block_size.x * gpu_grid_dim.x,
        gpu_block_size.y * gpu_grid_dim.y,
        gpu_block_size.z * gpu_grid_dim.z,
    };

    size_t nthreads_total = nthreads.x * nthreads.y * nthreads.z;
    xpu::hd_buffer<int> thread_idx{nthreads_total * 3};
    xpu::hd_buffer<int> block_dim{nthreads_total * 3};
    xpu::hd_buffer<int> block_idx{nthreads_total * 3};
    xpu::hd_buffer<int> grid_dim{nthreads_total * 3};

    xpu::grid exec_grid = xpu::grid::n_threads(nthreads);
    switch (gpu_block_size.ndims()) {
    case 1:
        xpu::run_kernel<get_thread_idx_1d>(exec_grid, thread_idx.d(), block_dim.d(), block_idx.d(), grid_dim.d());
        break;
    case 2:
        xpu::run_kernel<get_thread_idx_2d>(exec_grid, thread_idx.d(), block_dim.d(), block_idx.d(), grid_dim.d());
        break;
    case 3:
        xpu::run_kernel<get_thread_idx_3d>(exec_grid, thread_idx.d(), block_dim.d(), block_idx.d(), grid_dim.d());
        break;
    default:
        FAIL();
        break;
    }
    xpu::copy(thread_idx, xpu::device_to_host);
    xpu::copy(block_dim, xpu::device_to_host);
    xpu::copy(block_idx, xpu::device_to_host);
    xpu::copy(grid_dim, xpu::device_to_host);

    xpu::dim exp_block_dim = (xpu::active_driver() == xpu::cpu ? xpu::dim{1, 1, 1} : gpu_block_size);
    xpu::dim exp_grid_dim;
    exec_grid.get_compute_grid(exp_block_dim, exp_grid_dim);
    for (int i = 0; i < nthreads.x; i++) {
        for (int j = 0; j < nthreads.y; j++) {
            for (int k = 0; k < nthreads.z; k++) {

                xpu::dim exp_thread_idx {
                    i % exp_block_dim.x,
                    j % exp_block_dim.y,
                    k % exp_block_dim.z,
                };

                xpu::dim exp_block_idx {
                    i / exp_block_dim.x,
                    j / exp_block_dim.y,
                    k / exp_block_dim.z,
                };

                int threadsPerBlock = exp_block_dim.x * exp_block_dim.y * exp_block_dim.z;
                int threadIdxInBlock = exp_block_dim.x * exp_block_dim.y * exp_thread_idx.z + exp_block_dim.x * exp_thread_idx.y + exp_thread_idx.x;
                int blockNumInGrid = exp_grid_dim.x * exp_grid_dim.y * exp_block_idx.z + exp_grid_dim.x * exp_block_idx.y + exp_block_idx.x;

                int linear_id = blockNumInGrid * threadsPerBlock + threadIdxInBlock;
                linear_id *= 3;

                EXPECT_EQ(thread_idx.h()[linear_id + 0], exp_thread_idx.x);
                EXPECT_EQ(thread_idx.h()[linear_id + 1], exp_thread_idx.y);
                EXPECT_EQ(thread_idx.h()[linear_id + 2], exp_thread_idx.z);
                EXPECT_EQ(block_dim.h()[linear_id + 0], exp_block_dim.x);
                EXPECT_EQ(block_dim.h()[linear_id + 1], exp_block_dim.y);
                EXPECT_EQ(block_dim.h()[linear_id + 2], exp_block_dim.z);
                EXPECT_EQ(block_idx.h()[linear_id + 0], exp_block_idx.x);
                EXPECT_EQ(block_idx.h()[linear_id + 1], exp_block_idx.y);
                EXPECT_EQ(block_idx.h()[linear_id + 2], exp_block_idx.z);
                EXPECT_EQ(grid_dim.h()[linear_id + 0], exp_grid_dim.x);
                EXPECT_EQ(grid_dim.h()[linear_id + 1], exp_grid_dim.y);
                EXPECT_EQ(grid_dim.h()[linear_id + 2], exp_grid_dim.z);
            }
        }
    }

}

TEST(XPUTest, CanStartKernel1D) {
    test_thread_position(xpu::dim{128}, xpu::dim{4});
}

TEST(XPUTest, CanStartKernel2D) {
    test_thread_position(xpu::dim{32, 8}, xpu::dim{2, 2});
}

TEST(XPUTest, CanStartkernel3D) {
    test_thread_position(xpu::dim{32, 8, 2}, xpu::dim{2, 2, 2});
}

TEST(XPUTest, CanCallDeviceFuncs) {
    xpu::hd_buffer<variant> buf{NUM_DEVICE_FUNCS};
    xpu::memset(buf, 0);

    xpu::run_kernel<test_device_funcs>(xpu::grid::n_blocks(1), buf.d());
    xpu::copy(buf, xpu::device_to_host);

    variant *b = buf.h();

    EXPECT_FLOAT_EQ(b[ABS].f, 1.f);
    EXPECT_FLOAT_EQ(b[ACOS].f, xpu::pi());
    EXPECT_FLOAT_EQ(b[ACOSH].f, 0.f);
    EXPECT_FLOAT_EQ(b[ASIN].f, xpu::pi_2());
    EXPECT_FLOAT_EQ(b[ASINH].f, 0.88137358f);
    EXPECT_FLOAT_EQ(b[ATAN2].f, xpu::pi_4());
    EXPECT_FLOAT_EQ(b[ATAN].f, xpu::pi_4());
    EXPECT_FLOAT_EQ(b[ATANH].f, 1.4722193f);
    EXPECT_FLOAT_EQ(b[CBRT].f, 9.f);
    EXPECT_FLOAT_EQ(b[CEIL].f, 3.f);
    EXPECT_FLOAT_EQ(b[COPYSIGN].f, -1.f);
    EXPECT_FLOAT_EQ(b[COS].f, 0.5f);
    EXPECT_FLOAT_EQ(b[COSH].f, 1.5430807f);
    EXPECT_FLOAT_EQ(b[COSPI].f, -0.98902732f);
    EXPECT_FLOAT_EQ(b[ERF].f, 0.84270079294971f);
    EXPECT_FLOAT_EQ(b[ERFC].f, 1.f);
    EXPECT_FLOAT_EQ(b[EXP2].f, 16.f);
    EXPECT_FLOAT_EQ(b[EXP].f, 7.38905609893065f);
    EXPECT_FLOAT_EQ(b[EXPM1].f, 1.7182819f);
    EXPECT_FLOAT_EQ(b[FDIM].f, 3.f);
    EXPECT_FLOAT_EQ(b[FLOOR].f, 2.f);
    EXPECT_FLOAT_EQ(b[FMA].f, 10.f);
    EXPECT_FLOAT_EQ(b[FMOD].f, 2.1f);
    EXPECT_FLOAT_EQ(b[HYPOT].f, xpu::sqrt2());
    EXPECT_EQ(b[ILOGB].i, 6);
    EXPECT_TRUE(b[ISFINITE].b);
    EXPECT_TRUE(b[ISINF].b);
    EXPECT_TRUE(b[ISNAN].b);
    EXPECT_FLOAT_EQ(b[J0].f, 0.76519775f);
    EXPECT_FLOAT_EQ(b[J1].f, 0.4400506f);
    EXPECT_NEAR(b[JN].f, 0.11490349f, 0.0000001f);
    EXPECT_FLOAT_EQ(b[LDEXP].f, 0.4375f);
    EXPECT_EQ(b[LLRINT].ll, 2);
    EXPECT_EQ(b[LLROUND].ll, 2);
    EXPECT_FLOAT_EQ(b[LOG].f, 0.f);
    EXPECT_FLOAT_EQ(b[LOG10].f, 3.f);
    EXPECT_FLOAT_EQ(b[LOG1P].f, 0.f);
    EXPECT_FLOAT_EQ(b[LOG2].f, 5.f);
    EXPECT_FLOAT_EQ(b[LOGB].f, 6.f);
    EXPECT_EQ(b[LRINT].ll, 2);
    EXPECT_EQ(b[LROUND].ll, 2);
    EXPECT_FLOAT_EQ(b[MAX].f, 1.f);
    EXPECT_FLOAT_EQ(b[MIN].f, -1.f);
    EXPECT_EQ(b[NEARBYINT].f, 2.f);
    EXPECT_FLOAT_EQ(b[NORM].f, std::sqrt(5.f));
    EXPECT_FLOAT_EQ(b[NORM3D].f, std::sqrt(29.f));
    EXPECT_FLOAT_EQ(b[NORM4D].f, std::sqrt(54.f));
    EXPECT_FLOAT_EQ(b[POW].f, 27.f);
    EXPECT_FLOAT_EQ(b[RCBRT].f, 1.f / 3.f);
    EXPECT_FLOAT_EQ(b[REMAINDER].f, -0.9f);
    EXPECT_FLOAT_EQ(b[REMQUO_REM].f, 1.3f);
    EXPECT_EQ(b[REMQUO_QUO].i, 2);
    EXPECT_FLOAT_EQ(b[RHYPOT].f, 1.f / std::hypotf(2.f, 3.f));
    EXPECT_FLOAT_EQ(b[RINT].f, 2.f);
    EXPECT_FLOAT_EQ(b[RNORM].f, 1.f / std::sqrt(5.f));
    EXPECT_FLOAT_EQ(b[RNORM3D].f, 1.f / std::sqrt(29.f));
    EXPECT_FLOAT_EQ(b[RNORM4D].f, 1.f / std::sqrt(54.f));
    EXPECT_FLOAT_EQ(b[ROUND].f, 3.f);
    EXPECT_FLOAT_EQ(b[RSQRT].f,  0.5f);
    EXPECT_NEAR(b[SINCOS_SIN].f, 0.f, 0.0000001f);
    EXPECT_FLOAT_EQ(b[SINCOS_COS].f, -1.f);
    EXPECT_NEAR(b[SINCOSPI_SIN].f, 0.f, 0.0000001f);
    EXPECT_FLOAT_EQ(b[SINCOSPI_COS].f, -1.f);
    EXPECT_NEAR(b[SIN].f, 0.f, 0.0000001f);
    EXPECT_FLOAT_EQ(b[SINH].f, 1.1752012f);
    EXPECT_NEAR(b[SINPI].f, 0.f, 0.0000001f);
    EXPECT_FLOAT_EQ(b[SQRT].f, 8.f);
    EXPECT_FLOAT_EQ(b[TAN].f, 1.f);
    EXPECT_FLOAT_EQ(b[TANH].f, 0.76159418f);
    EXPECT_FLOAT_EQ(b[TGAMMA].f, 362880);
    EXPECT_FLOAT_EQ(b[TRUNC].f, 2.f);
    EXPECT_FLOAT_EQ(b[Y0].f, 0.088256963f);
    EXPECT_FLOAT_EQ(b[Y1].f, -0.78121281f);
    EXPECT_FLOAT_EQ(b[YN].f, -1.6506826f);
}

TEST(XPUTest, CollectsTimingData) {
    constexpr int NRuns = 10;
    constexpr int NElems = 100000;

    xpu::hd_buffer<float> a{NElems};
    xpu::hd_buffer<float> b{NElems};
    xpu::hd_buffer<float> c{NElems};

    std::fill_n(a.h(), a.size(), 24);
    std::fill_n(b.h(), b.size(), 24);

    xpu::copy(a, xpu::host_to_device);
    xpu::copy(b, xpu::host_to_device);

    for (int i = 0; i < NRuns; i++) {
        xpu::run_kernel<vector_add_timing0>(xpu::grid::n_threads(NElems), a.d(), b.d(), c.d(), NElems);
        xpu::run_kernel<vector_add_timing1>(xpu::grid::n_threads(NElems), a.d(), b.d(), c.d(), NElems);
    }

    auto timings0 = xpu::get_timing<vector_add_timing0>();

    ASSERT_EQ(timings0.size(), NRuns);

    for (auto &t : timings0) {
        ASSERT_GT(t, 0.f);
    }

    auto timings1 = xpu::get_timing<vector_add_timing1>();

    ASSERT_EQ(timings1.size(), NRuns);

    for (auto &t : timings1) {
        ASSERT_GT(t, 0.f);
    }

    ASSERT_NE(timings0, timings1);
}

TEST(XPUTest, CanCallImageFunction) {
    xpu::driver_t driver;
    xpu::call<get_driver_type>(&driver);

    xpu::driver_t expected_driver = xpu::active_driver();

    ASSERT_EQ(driver, expected_driver);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    setenv("XPU_PROFILE", "1", 1); // always enable profiling in unittests
    xpu::initialize();
    return RUN_ALL_TESTS();
}
