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
    xpu::buffer<int *> buf{};
    xpu::buffer<key_value_t *> buf1{100, xpu::buf_pinned};
    xpu::buffer<key_value_t *> buf2{100, xpu::buf_device};
}

TEST(XPUTest, CanGetDeviceFromPointer) {
    xpu::device_prop active_device{xpu::device::active()};
    bool is_cpu = active_device.backend() == xpu::driver_t::cpu;

    int h = 0;
    xpu::ptr_prop prop{&h};
    ASSERT_EQ(prop.ptr(), &h);
    ASSERT_EQ(prop.backend(), xpu::cpu);
    ASSERT_EQ(prop.device().device_nr(), 0);
    ASSERT_EQ(prop.type(), xpu::mem_type::host);
    ASSERT_TRUE(prop.is_host());

    {
        xpu::buffer<int> hdbuf{1, xpu::buf_io};
        xpu::buffer_prop bprop{hdbuf};
        prop = xpu::ptr_prop{bprop.h_ptr()};
        ASSERT_NE(prop.ptr(), nullptr);
        ASSERT_EQ(prop.backend(), active_device.backend());
        ASSERT_EQ(prop.type(), (is_cpu ? xpu::mem_type::host : xpu::mem_type::pinned));
        ASSERT_TRUE(prop.is_host());
        prop = xpu::ptr_prop{bprop.d_ptr()};
        ASSERT_NE(prop.ptr(), nullptr);
        ASSERT_EQ(prop.backend(), active_device.backend());
        ASSERT_EQ(prop.type(), (is_cpu ? xpu::mem_type::host : xpu::mem_type::device));
        ASSERT_EQ(prop.is_host(), is_cpu);
    }

    {
        xpu::buffer<int> dbuf{1, xpu::buf_device};
        xpu::buffer_prop bprop{dbuf};
        prop = xpu::ptr_prop{bprop.d_ptr()};
        ASSERT_NE(prop.ptr(), nullptr);
        ASSERT_EQ(prop.backend(), active_device.backend());
        ASSERT_EQ(prop.type(), (is_cpu ? xpu::mem_type::host : xpu::mem_type::device));
        ASSERT_EQ(bprop.h_ptr(), nullptr);
    }

    {
        xpu::buffer<int> hbuf{1, xpu::buf_pinned};
        xpu::buffer_prop bprop{hbuf};
        prop = xpu::ptr_prop{bprop.h_ptr()};
        ASSERT_NE(prop.ptr(), nullptr);
        ASSERT_EQ(prop.backend(), active_device.backend());
        ASSERT_EQ(prop.type(), (is_cpu ? xpu::mem_type::host : xpu::mem_type::pinned));
        ASSERT_TRUE(prop.is_host());
    }
}

TEST(XPUTest, CanConvertTypenamesToString) {
    ASSERT_STREQ(xpu::detail::type_name<int>(), "int");
    ASSERT_STREQ(xpu::detail::type_name<xpu::device_prop>(), "xpu::device_prop");
}

TEST(XPUTest, CanWriteBufferToCMem) {
    xpu::buffer<int> buf{1, xpu::buf_device};
    xpu::set<cmem_buffer>(buf);
}

TEST(XPUTest, HostBufferIsAccessibleFromDevice) {
    xpu::buffer<int> buf{1, xpu::buf_pinned};
    xpu::buffer<int> x{};
    *buf = 69;
    xpu::queue q;
    q.launch<buffer_access>(xpu::n_threads(1), x, buf);
    q.wait();
    ASSERT_EQ(*buf, 42);
}

TEST(XPUTest, IoBufferIsAccessibleFromDevice) {
    xpu::buffer<int> buf{1, xpu::buf_io};
    xpu::buffer<int> x{};

    xpu::queue q;
    q.launch<buffer_access>(xpu::n_threads(1), x, buf);
    q.copy(buf, xpu::d2h);
    q.wait();

    auto bview = xpu::h_view{buf};
    ASSERT_EQ(bview[0], 42);
    xpu::buffer_prop bprop{buf};
    ASSERT_EQ(bprop.h_ptr(), bview.begin());
}

TEST(XPUTest, IoBufferCanCopyToHost) {
    int val = 69;
    xpu::buffer<int> buf{1, xpu::buf_io, &val};
    xpu::buffer<int> x{};

    xpu::queue q;
    q.launch<buffer_access>(xpu::n_threads(1), x, buf);
    q.copy(buf, xpu::d2h);
    q.wait();

    ASSERT_EQ(val, 42);
}

TEST(XPUTest, CanCopyAsyncDeviceToHost) {
    int val = 69;
    xpu::buffer<int> buf{1, xpu::buf_io, &val};
    xpu::buffer<int> x{};

    xpu::queue q;
    q.launch<buffer_access>(xpu::n_threads(1), x, buf);
    q.copy(buf, xpu::d2h);
    q.wait();
    ASSERT_EQ(val, 42);
}

TEST(XPUTest, CanAllocateStackMemory) {
    xpu::stack_alloc(1024 * 1024);
    xpu::buffer<int> buf{1, xpu::buf_stack};
    xpu::buffer<int> x{};

    xpu::buffer_prop bprop{buf};
    ASSERT_EQ(bprop.type(), xpu::buf_stack);
    ASSERT_EQ(bprop.size(), 1);
    ASSERT_EQ(bprop.h_ptr(), nullptr);
    ASSERT_NE(bprop.d_ptr(), nullptr);
    // Ensure 256 byte alignment
    ASSERT_EQ(reinterpret_cast<uintptr_t>(bprop.d_ptr()) % 256, 0);

    xpu::queue q;
    q.launch<buffer_access>(xpu::n_threads(1), x, buf);

    int result;
    q.copy(buf.get(), &result, 1);
    q.wait();

    ASSERT_EQ(result, 42);

    // Test if we can push / pop on the stack
    void *head = buf.get();

    xpu::buffer<int> buf2{1, xpu::buf_stack};
    ASSERT_GT(buf2.get(), head);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(buf2.get()) % 256, 0);

    xpu::buffer<int> buf3{1, xpu::buf_stack};
    ASSERT_GT(buf3.get(), buf2.get());
    ASSERT_EQ(reinterpret_cast<uintptr_t>(buf3.get()) % 256, 0);

    xpu::stack_pop(buf3.get());
    xpu::buffer<int> buf4{1, xpu::buf_stack};
    ASSERT_EQ(buf4.get(), buf3.get());

    xpu::stack_pop();
    xpu::buffer<int> buf5{1, xpu::buf_stack};
    ASSERT_EQ(buf5.get(), head);
}

TEST(XPUTest, CanRunVectorAdd) {
    constexpr int NElems = 100;

    std::vector<float> hx(NElems, 8);
    std::vector<float> hy(NElems, 8);

    float *dx = xpu::malloc_device<float>(NElems);
    float *dy = xpu::malloc_device<float>(NElems);
    float *dz = xpu::malloc_device<float>(NElems);

    xpu::queue q;
    q.copy(hx.data(), dx, NElems);
    q.copy(hy.data(), dy, NElems);
    q.launch<vector_add>(xpu::n_threads(NElems), dx, dy, dz, NElems);

    std::vector<float> hz(NElems);
    q.copy(dz, hz.data(), NElems);
    q.wait();

    for (auto &x: hz) {
        ASSERT_EQ(16, x);
    }

    xpu::free(dx);
    xpu::free(dy);
    xpu::free(dz);
}

TEST(XPUTest, CanRunVectorAddQueue) {
    constexpr int NElems = 100;

    xpu::buffer<float> xbuf{NElems, xpu::buf_io};
    xpu::buffer<float> ybuf{NElems, xpu::buf_io};
    xpu::buffer<float> zbuf{NElems, xpu::buf_io};

    xpu::h_view x{xbuf};
    xpu::h_view y{ybuf};
    for (int i = 0; i < NElems; ++i) {
        x[i] = 8;
        y[i] = 8;
    }

    xpu::queue q{};

    q.copy(xbuf, xpu::h2d);
    q.copy(ybuf, xpu::h2d);
    q.launch<vector_add>(xpu::n_threads(NElems), xbuf.get(), ybuf.get(), zbuf.get(), NElems);
    q.copy(zbuf, xpu::d2h);
    q.wait();

    xpu::h_view z{zbuf};
    for (int i = 0; i < NElems; ++i) {
        ASSERT_EQ(16, z[i]);
    }

}

TEST(XPUTest, CanSortStruct) {

    // GTEST_SKIP();

    constexpr size_t NElems = 1000000;

    std::mt19937 gen{1337};
    std::uniform_int_distribution<unsigned int> dist{0, 100000000};

    key_value_t *ditems = xpu::malloc_device<key_value_t>(NElems);
    key_value_t *buf = xpu::malloc_device<key_value_t>(NElems);
    key_value_t **dst = xpu::malloc_device<key_value_t *>(1);


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


    xpu::queue q;
    q.copy(items.data(), ditems, NElems);
    q.launch<sort_struct>(xpu::n_blocks(1), ditems, NElems, buf, dst);

    key_value_t *hdst = nullptr;
    q.copy(dst, &hdst, 1);
    q.wait();
    q.copy(hdst, items.data(), NElems);
    q.wait();

    // for (auto &x : items) {
    //     std::cout << x.key << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "----------------" << std::endl;
    // for (auto &x : itemsSorted) {
    //     std::cout << x.key << " ";
    // }
    // std::cout << std::endl;


    for (size_t i = 0; i < NElems; i++) {
        EXPECT_EQ(items[i].key, itemsSorted[i].key) << " with i = " << i;
        ASSERT_EQ(items[i].value, itemsSorted[i].value);
    }

}

TEST(XPUTest, CanSortFloatsShort) {

    // GTEST_SKIP();

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

    float *ditems = xpu::malloc_device<float>(NElems);
    float *buf = xpu::malloc_device<float>(NElems);
    float **dst = xpu::malloc_device<float *>(1);

    xpu::queue q;
    q.copy(items.data(), ditems, NElems);
    q.launch<sort_float>(xpu::n_blocks(1), ditems, NElems, buf, dst);

    float *hdst = nullptr;
    q.copy(dst, &hdst, 1);
    q.wait();
    q.copy(hdst, items.data(), NElems);
    q.wait();

    // for (auto &x : items) {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Sorted" << std::endl;
    // for (auto &x : itemsSorted) {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;

    for (size_t i = 0; i < NElems; i++) {
        ASSERT_EQ(items[i], itemsSorted[i]);
    }

}

template<typename K>
void testMergeKernel(size_t M, size_t N) {
    xpu::buffer<float> a{M, xpu::buf_io};
    xpu::buffer<float> b{N, xpu::buf_io};
    xpu::buffer<float> dst{M + N, xpu::buf_io};

    std::mt19937 gen{1337};
    std::uniform_real_distribution<float> dist{0, 100000};

    xpu::h_view a_h{a};
    for (size_t i = 0; i < M; i++) {
        a_h[i] = dist(gen);
    }
    std::sort(&a_h[0], &a_h[0] + a_h.size());

    xpu::h_view b_h{b};
    for (size_t i = 0; i < N; i++) {
        b_h[i] = dist(gen);
    }
    std::sort(&b_h[0], &b_h[0] + b_h.size());

    xpu::queue q;

    q.copy(a, xpu::h2d);
    q.copy(b, xpu::h2d);
    q.launch<K>(xpu::n_blocks(1), a.get(), a_h.size(), b.get(), b_h.size(), dst.get());
    q.copy(dst, xpu::d2h);
    q.wait();

    xpu::h_view h{dst};
    ASSERT_EQ(h.size(), a_h.size() + b_h.size());
    bool isSorted = true;
    for (size_t i = 1; i < h.size(); i++) {
        isSorted &= (h[i-1] <= h[i]);
    }

    if (!isSorted) {
        for (size_t i = 0; i < h.size(); i++) {
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
#ifdef DONT_TEST_BLOCK_SORT
    GTEST_SKIP();
#endif
    testMergeKernel<merge>(512, 512);
}

TEST(XPUTest, CanMergeUnevenNumberOfItems) {
#ifdef DONT_TEST_BLOCK_SORT
    GTEST_SKIP();
#endif
    testMergeKernel<merge>(610, 509);
}

TEST(XPUTest, CanMergeWithOneItemPerThread) {
#ifdef DONT_TEST_BLOCK_SORT
    GTEST_SKIP();
#endif
    testMergeKernel<merge_single>(512, 512);
}

TEST(XPUTest, CanRunBlockScan) {
#ifdef DONT_TEST_BLOCK_FUNCS
    GTEST_SKIP();
#endif

    size_t blockSize = xpu::device::active().backend() == xpu::cpu ? 1 : 64;

    xpu::buffer<int> incl{blockSize, xpu::buf_io};
    xpu::buffer<int> excl{blockSize, xpu::buf_io};

    xpu::queue q;
    q.launch<block_scan>(xpu::n_blocks(1), incl.get(), excl.get());
    q.copy(incl, xpu::d2h);
    q.copy(excl, xpu::d2h);
    q.wait();

    int inclSum = 1;
    int exclSum = 0;

    xpu::h_view incl_h{incl};
    xpu::h_view excl_h{excl};
    for (size_t i = 0; i < blockSize; i++) {
        ASSERT_EQ(incl_h[i], inclSum);
        ASSERT_EQ(excl_h[i], exclSum);
        inclSum++;
        exclSum++;
    }
}

TEST(XPUTest, CanRunBlockReduce) {
#ifdef DONT_TEST_BLOCK_FUNCS
    GTEST_SKIP();
#endif
    // Skip on SYCL backend, because compilation fails with reduction for some reason...
    if (xpu::device::active().backend() == xpu::sycl) {
        GTEST_SKIP();
    }
    size_t blockSize = xpu::device::active().backend() == xpu::cpu ? 1 : 64;

    xpu::buffer<int> sum{1, xpu::buf_io};
    xpu::buffer<int> any{1, xpu::buf_io};

    xpu::queue q;
    q.launch<block_reduce>(xpu::n_blocks(1), sum.get(), any.get());
    q.copy(sum, xpu::d2h);
    q.copy(any, xpu::d2h);
    q.wait();

    xpu::h_view sum_h{sum};
    xpu::h_view any_h{any};

    ASSERT_EQ(sum_h[0], blockSize);
    ASSERT_EQ(any_h[0], 1);
}

TEST(XPUTest, CanSetAndReadCMem) {
    float3_ orig{1, 2, 3};
    xpu::buffer<float3_> out{1, xpu::buf_io};

    xpu::set<test_constant0>(orig);

    xpu::queue q;
    q.launch<access_cmem_single>(xpu::n_threads(1), out.get());
    q.copy(out, xpu::d2h);

    float3_ result = xpu::h_view(out)[0];
    EXPECT_EQ(orig.x, result.x);
    EXPECT_EQ(orig.y, result.y);
    EXPECT_EQ(orig.z, result.z);
}

TEST(XPUTest, CanSetAndReadCMemMultiple) {
    float3_ orig{1, 2, 3};
    xpu::buffer<float3_> out0{1, xpu::buf_io};

    double orig1 = 42;
    xpu::buffer<double> out1{1, xpu::buf_io};

    float orig2 = 1337;
    xpu::buffer<float> out2{1, xpu::buf_io};

    xpu::set<test_constant0>(orig);
    xpu::set<test_constant1>(orig1);
    xpu::set<test_constant2>(orig2);

    xpu::queue q;
    q.launch<access_cmem_multiple>(xpu::n_threads(1), out0.get(), out1.get(), out2.get());
    q.copy(out0, xpu::d2h);
    q.copy(out1, xpu::d2h);
    q.copy(out2, xpu::d2h);
    q.wait();

    {
        float3_ result = xpu::h_view(out0)[0];
        EXPECT_EQ(orig.x, result.x);
        EXPECT_EQ(orig.y, result.y);
        EXPECT_EQ(orig.z, result.z);
    }
    {
        double result = xpu::h_view(out1)[0];
        EXPECT_EQ(orig1, result);
    }
    {
        float result = xpu::h_view(out2)[0];
        EXPECT_EQ(orig2, result);
    }
}

void test_thread_position(xpu::dim gpu_block_size, xpu::dim gpu_grid_dim) {

    xpu::dim nthreads{
        gpu_block_size.x * gpu_grid_dim.x,
        gpu_block_size.y * gpu_grid_dim.y,
        gpu_block_size.z * gpu_grid_dim.z,
    };

    size_t nthreads_total = nthreads.x * nthreads.y * nthreads.z;
    xpu::buffer<int> thread_idx{nthreads_total * 3, xpu::buf_io};
    xpu::buffer<int> block_dim{nthreads_total * 3, xpu::buf_io};
    xpu::buffer<int> block_idx{nthreads_total * 3, xpu::buf_io};
    xpu::buffer<int> grid_dim{nthreads_total * 3, xpu::buf_io};

    xpu::grid exec_grid = xpu::n_threads(nthreads);
    xpu::queue q;

    switch (gpu_block_size.ndims()) {
    case 1:
        q.launch<get_thread_idx_1d>(exec_grid, thread_idx.get(), block_dim.get(), block_idx.get(), grid_dim.get());
        break;
    case 2:
        q.launch<get_thread_idx_2d>(exec_grid, thread_idx.get(), block_dim.get(), block_idx.get(), grid_dim.get());
        break;
    case 3:
        q.launch<get_thread_idx_3d>(exec_grid, thread_idx.get(), block_dim.get(), block_idx.get(), grid_dim.get());
        break;
    default:
        FAIL();
        break;
    }
    q.copy(thread_idx, xpu::d2h);
    q.copy(block_dim, xpu::d2h);
    q.copy(block_idx, xpu::d2h);
    q.copy(grid_dim, xpu::d2h);

    xpu::dim exp_block_dim = (xpu::device::active().backend() == xpu::cpu ? xpu::dim{1, 1, 1} : gpu_block_size);
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

                xpu::h_view thread_idx_h = xpu::h_view(thread_idx);
                xpu::h_view block_dim_h = xpu::h_view(block_dim);
                xpu::h_view block_idx_h = xpu::h_view(block_idx);
                xpu::h_view grid_dim_h = xpu::h_view(grid_dim);
                ASSERT_EQ(thread_idx_h[linear_id + 0], exp_thread_idx.x);
                ASSERT_EQ(thread_idx_h[linear_id + 1], exp_thread_idx.y);
                ASSERT_EQ(thread_idx_h[linear_id + 2], exp_thread_idx.z);
                ASSERT_EQ(block_dim_h[linear_id + 0], exp_block_dim.x);
                ASSERT_EQ(block_dim_h[linear_id + 1], exp_block_dim.y);
                ASSERT_EQ(block_dim_h[linear_id + 2], exp_block_dim.z);
                ASSERT_EQ(block_idx_h[linear_id + 0], exp_block_idx.x);
                ASSERT_EQ(block_idx_h[linear_id + 1], exp_block_idx.y);
                ASSERT_EQ(block_idx_h[linear_id + 2], exp_block_idx.z);
                ASSERT_EQ(grid_dim_h[linear_id + 0], exp_grid_dim.x);
                ASSERT_EQ(grid_dim_h[linear_id + 1], exp_grid_dim.y);
                ASSERT_EQ(grid_dim_h[linear_id + 2], exp_grid_dim.z);
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
    xpu::buffer<variant> buf{NUM_DEVICE_FUNCS, xpu::buf_io};

    xpu::queue q;
    q.memset(buf, 0);
    q.launch<test_device_funcs>(xpu::n_blocks(1), buf.get());
    q.copy(buf, xpu::d2h);
    q.wait();

    xpu::h_view b{buf};

    EXPECT_FLOAT_EQ(b[ABS].f, 1.f);
    EXPECT_FLOAT_EQ(b[ACOS].f, xpu::pi());
    EXPECT_FLOAT_EQ(b[ACOSH].f, 0.f);
    EXPECT_FLOAT_EQ(b[ACOSPI].f, 1.f);
    EXPECT_FLOAT_EQ(b[ASIN].f, xpu::pi_2());
    EXPECT_FLOAT_EQ(b[ASINH].f, 0.88137358f);
    EXPECT_FLOAT_EQ(b[ASINPI].f, 0.5f);
    EXPECT_FLOAT_EQ(b[ATAN2].f, xpu::pi_4());
    EXPECT_FLOAT_EQ(b[ATAN].f, xpu::pi_4());
    EXPECT_FLOAT_EQ(b[ATANH].f, 1.4722193f);
    EXPECT_FLOAT_EQ(b[ATANPI].f, 0.25f);
    EXPECT_FLOAT_EQ(b[ATAN2PI].f, 0.25f);
    EXPECT_FLOAT_EQ(b[CBRT].f, 9.f);
    EXPECT_FLOAT_EQ(b[CEIL].f, 3.f);
    EXPECT_FLOAT_EQ(b[COPYSIGN].f, -1.f);
    EXPECT_NEAR(b[COS].f, 0.5f, 2e-5f);
    EXPECT_FLOAT_EQ(b[COSH].f, 1.5430807f);
    EXPECT_FLOAT_EQ(b[COSPI].f, -0.98902732f);
    EXPECT_FLOAT_EQ(b[ERF].f, 0.84270079294971f);
    EXPECT_FLOAT_EQ(b[ERFC].f, 1.f);
    EXPECT_FLOAT_EQ(b[EXP2].f, 16.f);
    EXPECT_NEAR(b[EXP10].f, 10000.f, 0.4f); // sycl::exp10 gives 9999.6289
    EXPECT_NEAR(b[EXP].f, 7.389f, 0.0001f); // sycl::exp gives 7.3889837
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
    EXPECT_FLOAT_EQ(b[NORM3D].f, std::sqrt(29.f));
    EXPECT_FLOAT_EQ(b[NORM4D].f, std::sqrt(54.f));
    EXPECT_FLOAT_EQ(b[POW].f, 27.f);
    EXPECT_FLOAT_EQ(b[RCBRT].f, 1.f / 3.f);
    EXPECT_FLOAT_EQ(b[REMAINDER].f, -0.9f);
    EXPECT_FLOAT_EQ(b[REMQUO_REM].f, 1.3f);
    EXPECT_EQ(b[REMQUO_QUO].i, 2);
    EXPECT_FLOAT_EQ(b[RHYPOT].f, 1.f / std::hypotf(2.f, 3.f));
    EXPECT_FLOAT_EQ(b[RINT].f, 2.f);
    EXPECT_FLOAT_EQ(b[RNORM3D].f, 1.f / std::sqrt(29.f));
    EXPECT_FLOAT_EQ(b[RNORM4D].f, 1.f / std::sqrt(54.f));
    EXPECT_FLOAT_EQ(b[ROUND].f, 3.f);
    EXPECT_NEAR(b[RSQRT].f,  0.5f, 0.0002f); // sycl::rsqrt gives 0.49987793
    EXPECT_NEAR(b[SINCOS_SIN].f, 0.f, 0.0000001f);
    EXPECT_NEAR(b[SINCOS_COS].f, -1.f, 0.00001); // sycl::sincos gives -0.99999827
    EXPECT_NEAR(b[SINCOSPI_SIN].f, 0.f, 0.0000001f);
    EXPECT_NEAR(b[SINCOSPI_COS].f, -1.f, 0.00001f); // sycl::sincospi gives -0.99999827
    EXPECT_NEAR(b[SIN].f, 0.f, 0.0000001f);
    EXPECT_FLOAT_EQ(b[SINH].f, 1.1752012f);
    EXPECT_NEAR(b[SINPI].f, 0.f, 0.0000001f);
    EXPECT_FLOAT_EQ(b[SQRT].f, 8.f);
    EXPECT_FLOAT_EQ(b[TAN].f, 1.f);
    EXPECT_FLOAT_EQ(b[TANH].f, 0.76159418f);
    EXPECT_FLOAT_EQ(b[TGAMMA].f, 362880);
    EXPECT_FLOAT_EQ(b[TRUNC].f, 2.f);
}

TEST(XPUTest, CanRunTemplatedKernels) {
    xpu::buffer<int> a{1, xpu::buf_io};

    xpu::queue q;
    q.launch<templated_kernel<0>>(xpu::n_threads(1), a.get());
    q.copy(a, xpu::d2h);
    q.wait();
    ASSERT_EQ(xpu::h_view{a}[0], 0);

    q.launch<templated_kernel<1>>(xpu::n_threads(1), a.get());
    q.copy(a, xpu::d2h);
    q.wait();
    ASSERT_EQ(xpu::h_view{a}[0], 1);

    q.launch<templated_kernel<42>>(xpu::n_threads(1), a.get());
    q.copy(a, xpu::d2h);
    q.wait();
    ASSERT_EQ(xpu::h_view{a}[0], 42);
}

TEST(XPUTest, CollectsTimingData) {
    constexpr int NRuns = 10;
    constexpr int NElems = 100000;

    xpu::buffer<float> a{NElems, xpu::buf_io};
    xpu::buffer<float> b{NElems, xpu::buf_io};
    xpu::buffer<float> c{NElems, xpu::buf_io};

    xpu::queue q;

    xpu::push_timer("test");

    xpu::h_view h_a{a};
    std::fill(h_a.begin(), h_a.end(), 24.f);
    xpu::h_view h_b{b};
    std::fill(h_b.begin(), h_b.end(), 24.f);

    q.copy(a, xpu::h2d);
    q.copy(b, xpu::h2d);

    for (int i = 0; i < NRuns; i++) {
        q.launch<vector_add_timing0>(xpu::n_threads(NElems), a.get(), b.get(), c.get(), NElems);
        q.launch<vector_add_timing1>(xpu::n_threads(NElems), a.get(), b.get(), c.get(), NElems);
    }

    xpu::timings ts = xpu::pop_timer();

    ASSERT_EQ(std::string{ts.name()}, "test");
    ASSERT_TRUE(ts.has_details());
    ASSERT_EQ(ts.memset(), 0);
    ASSERT_EQ(ts.copy(xpu::d2h), 0);
    if (xpu::device::active().backend() != xpu::cpu) {
        ASSERT_GT(ts.copy(xpu::h2d), 0);
    }
    ASSERT_GT(ts.wall(), 0);

    xpu::kernel_timings timings0 = ts.kernel<vector_add_timing0>();
    ASSERT_EQ(timings0.name(), "vector_add_timing0");
    ASSERT_EQ(timings0.times().size(), NRuns);

    double total = 0;
    for (auto &t : timings0.times()) {
        ASSERT_GT(t, 0.f);
        total += t;
    }
    ASSERT_FLOAT_EQ(timings0.total(), total);

    auto timings1 = ts.kernel<vector_add_timing1>();
    ASSERT_EQ(timings1.name(), "vector_add_timing1");
    ASSERT_EQ(timings1.times().size(), NRuns);

    for (auto &t : timings1.times()) {
        ASSERT_GT(t, 0.f);
    }

    ASSERT_FLOAT_EQ(ts.kernel_time(), timings0.total() + timings1.total());
}

TEST(XPUTest, CanCallImageFunction) {
    xpu::driver_t driver;
    xpu::call<get_driver_type>(&driver);

    xpu::driver_t expected_driver = xpu::device::active().backend();

    ASSERT_EQ(driver, expected_driver);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    xpu::settings settings{};
    settings.profile = true; // Always enable profiling for tests
    xpu::initialize(settings);
    int ret = RUN_ALL_TESTS();

    // Clean up buffer in constant memory.
    // The xpu runtime might not be available anymore
    // when cpu constant memory is destructed.
    xpu::buffer<int> a{};
    xpu::set<cmem_buffer>(a);

    return ret;
}
