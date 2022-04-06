#include "bench_device.h"

#include <xpu/host.h>

#include <algorithm>
#include <iomanip>
#include <memory>
#include <random>
#include <string>
#include <sstream>
#include <vector>

class benchmark {

public:
    virtual ~benchmark() {}

    virtual std::string name() = 0;
    virtual void setup() = 0;
    virtual void teardown() = 0;
    virtual void run() = 0;
    virtual size_t bytes() { return 0; }
    virtual std::vector<float> timings() = 0;

};

class benchmark_runner {

public:
    void add(benchmark *b) { benchmarks.emplace_back(b); }

    void run(int n) {
        for (auto &b : benchmarks) {
            run_benchmark(b.get(), n);
        }

        print_entry("Benchmark");
        print_entry("Min");
        print_entry("Max");
        print_entry("Median");
        std::cout << std::endl;
        for (auto &b : benchmarks) {
            print_results(b.get());
        }
    }

private:
    std::vector<std::unique_ptr<benchmark>> benchmarks;

    void run_benchmark(benchmark *b, int r) {
        std::cout << "Running benchmark '" << b->name() << "'" << std::endl;
        b->setup();

        for (int i = 0; i < r + 1; i++) {
            b->run();
        }

        b->teardown();
    }

    void print_results(benchmark *b) {
        std::vector<float> timings = b->timings();
        size_t bytes = b->bytes();

        timings.erase(timings.begin()); // discard warmup run
        std::sort(timings.begin(), timings.end());

        print_entry(b->name());

        float min = timings.front();
        float max = timings.back();
        float median = timings[timings.size() / 2];

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);

        ss << min << "ms";
        if (bytes > 0) {
            ss << " (" << gb_s(bytes, min) << "GB/s)";
        }
        print_entry(ss.str());
        ss.str("");

        ss << max << "ms";
        if (bytes > 0) {
            ss << " (" << gb_s(bytes, max) << "GB/s)";
        }
        print_entry(ss.str());
        ss.str("");

        ss << median << "ms";
        if (bytes > 0) {
            ss << " (" << gb_s(bytes, median) << "GB/s)";
        }
        print_entry(ss.str());
        ss.str("");
        std::cout << std::endl;
    }

    void print_entry(std::string entry) const {
        std::cout << std::left << std::setw(25) << std::setfill(' ') << entry;
    }

    float gb_s(size_t bytes, float ms) const {
        return (bytes / (1000.f * 1000.f * 1000.f)) / (ms / 1000.f);
    }

};

template<typename Kernel>
class merge_bench : public benchmark {

private:
    static constexpr size_t elems_per_block = 32 * 32 * 200;
    static constexpr size_t n_blocks = 2000;
    static constexpr size_t buf_size = elems_per_block * n_blocks;

    xpu::hd_buffer<float> a;
    xpu::hd_buffer<float> b;
    xpu::hd_buffer<float> c;

public:
    std::string name() { return xpu::get_name<Kernel>(); }

    void setup() {
        a = xpu::hd_buffer<float>(buf_size);
        b = xpu::hd_buffer<float>(buf_size);
        c = xpu::hd_buffer<float>(buf_size * 2);

        std::mt19937 gen{42};

        std::uniform_real_distribution<float> dist{0, 1};

        float partial_sum = 0.f;
        auto rand_partial_sum = [&](){ partial_sum += dist(gen); return partial_sum; };

        std::generate(a.host(), a.host()+buf_size, rand_partial_sum);

        partial_sum = 0.f;
        std::generate(b.host(), b.host()+buf_size, rand_partial_sum);
    }

    void teardown() {
        a.reset();
        b.reset();
        c.reset();
    }

    void run() {
        xpu::copy(a, xpu::host_to_device);
        xpu::copy(b, xpu::host_to_device);

        xpu::run_kernel<Kernel>(xpu::grid::n_blocks(n_blocks), a.device(), b.device(), elems_per_block, c.device());

        xpu::copy(c, xpu::device_to_host);
    }

    size_t bytes() { return elems_per_block * n_blocks * 2 * sizeof(float); }
    std::vector<float> timings() { return xpu::get_timing<Kernel>(); }

};

template<typename T>
constexpr size_t merge_bench<T>::elems_per_block;

template<typename Kernel>
class sort_bench : public benchmark {

private:
    static constexpr size_t elems_per_block = 32 * 32 * 200;
    static constexpr size_t n_blocks = 2000;
    static constexpr size_t buf_size = elems_per_block * n_blocks;

    xpu::hd_buffer<float> a;
    xpu::hd_buffer<float> b;
    xpu::hd_buffer<float *> dst;

public:
    std::string name() { return xpu::get_name<Kernel>(); }

    void setup() {
        a = xpu::hd_buffer<float>(buf_size);
        b = xpu::hd_buffer<float>(buf_size);
        dst = xpu::hd_buffer<float *>(n_blocks);

        std::mt19937 gen{1337};

        std::uniform_real_distribution<float> dist{0, 1000000};

        auto rand = [&](){ return dist(gen); };

        std::generate(a.host(), a.host()+buf_size, rand);
    }

    void teardown() {
        a.reset();
        b.reset();
        dst.reset();
    }

    void run() {
        xpu::copy(a, xpu::host_to_device);
        xpu::run_kernel<Kernel>(xpu::grid::n_blocks(n_blocks), a.device(), elems_per_block, b.device(), dst.device());
        xpu::copy(dst, xpu::device_to_host);
    }

    size_t bytes() { return elems_per_block * n_blocks * sizeof(float); }
    std::vector<float> timings() { return xpu::get_timing<Kernel>(); }

};

template<typename T>
constexpr size_t sort_bench<T>::elems_per_block;

int main() {
    setenv("XPU_PROFILE", "1", 1); // always enable profiling in benchmark

    xpu::initialize();

    benchmark_runner runner;

    runner.add(new sort_bench<sort_i1>{});
    // Parameters don't have an effect on cpu merge, so running one sort benchmark is enough
    if (xpu::active_driver() != xpu::cpu) {
        runner.add(new sort_bench<sort_i2>{});
        runner.add(new sort_bench<sort_i4>{});
        runner.add(new sort_bench<sort_i8>{});
        runner.add(new sort_bench<sort_i12>{});
        runner.add(new sort_bench<sort_i16>{});
        runner.add(new sort_bench<sort_i32>{});
        runner.add(new sort_bench<sort_i48>{});
        runner.add(new sort_bench<sort_i64>{});
    }

    runner.add(new merge_bench<merge_i4>{});
    // Parameters don't have an effect on cpu sort, so running one sort benchmark is enough
    if (xpu::active_driver() != xpu::cpu) {
        runner.add(new merge_bench<merge_i8>{});
        runner.add(new merge_bench<merge_i12>{});
        runner.add(new merge_bench<merge_i16>{});
        runner.add(new merge_bench<merge_i32>{});
        runner.add(new merge_bench<merge_i48>{});
        runner.add(new merge_bench<merge_i64>{});
    }

    runner.run(10);

    return 0;
}
