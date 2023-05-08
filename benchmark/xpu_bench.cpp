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
    virtual std::vector<double> timings() = 0;

protected:
    xpu::timings m_timings;

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
        std::vector<double> timings = b->timings();
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

    xpu::buffer<float> a;
    xpu::buffer<float> b;
    xpu::buffer<float> c;

public:
    std::string name() { return xpu::get_name<Kernel>(); }

    void setup() {
        a.reset(buf_size, xpu::buf_io);
        b.reset(buf_size, xpu::buf_io);
        c.reset(buf_size * 2, xpu::buf_io);

        std::mt19937 gen{42};

        std::uniform_real_distribution<float> dist{0, 1};

        float partial_sum = 0.f;
        auto rand_partial_sum = [&](){ partial_sum += dist(gen); return partial_sum; };

        xpu::h_view a_h{a};
        std::generate(a_h.begin(), a_h.end(), rand_partial_sum);

        partial_sum = 0.f;
        xpu::h_view b_h{b};
        std::generate(b_h.begin(), b_h.end(), rand_partial_sum);
    }

    void teardown() {
        a.reset();
        b.reset();
        c.reset();
    }

    void run() {
        xpu::push_timer(name());
        xpu::queue q;
        q.copy(a, xpu::h2d);
        q.copy(b, xpu::h2d);
        q.launch<Kernel>(xpu::n_blocks(n_blocks), a.get(), b.get(), elems_per_block, c.get());
        q.copy(c, xpu::d2h);
        q.wait();
        m_timings.merge(xpu::pop_timer());
    }

    size_t bytes() { return elems_per_block * n_blocks * 2 * sizeof(float); }
    std::vector<double> timings() { return m_timings.kernel<Kernel>().times(); }

};

template<typename T>
constexpr size_t merge_bench<T>::elems_per_block;

template<typename Kernel>
class sort_bench : public benchmark {

private:
    static constexpr size_t elems_per_block = 32 * 32 * 200;
    static constexpr size_t n_blocks = 2000;
    static constexpr size_t buf_size = elems_per_block * n_blocks;

    xpu::buffer<float> a;
    xpu::buffer<float> b;
    xpu::buffer<float *> dst;

public:
    std::string name() { return xpu::get_name<Kernel>(); }

    void setup() {
        a.reset(buf_size, xpu::buf_io);
        b.reset(buf_size, xpu::buf_device);
        dst.reset(n_blocks, xpu::buf_io);

        std::mt19937 gen{1337};

        std::uniform_real_distribution<float> dist{0, 1000000};

        xpu::h_view a_h{a};
        auto rand = [&](){ return dist(gen); };
        std::generate(a_h.begin(), a_h.end(), rand);
    }

    void teardown() {
        a.reset();
        b.reset();
        dst.reset();
    }

    void run() {
        xpu::push_timer(name());
        xpu::queue q;
        q.copy(a, xpu::h2d);
        q.launch<Kernel>(xpu::n_blocks(n_blocks), a.get(), elems_per_block, b.get(), dst.get());
        q.copy(dst, xpu::d2h);
        q.wait();
        m_timings.merge(xpu::pop_timer());
    }

    size_t bytes() { return elems_per_block * n_blocks * sizeof(float); }
    std::vector<double> timings() { return m_timings.kernel<Kernel>().times(); }

};

template<typename T>
constexpr size_t sort_bench<T>::elems_per_block;

int main() {
    xpu::settings settings;
    settings.profile = true;
    xpu::initialize(settings);

    xpu::device_prop props{xpu::device::active()};
    std::cout << props.name() << std::endl;

    benchmark_runner runner;

    runner.add(new sort_bench<sort<1>>{});
    // Parameters don't have an effect on cpu merge, so running one sort benchmark is enough
    if (xpu::device::active().backend() != xpu::cpu) {
        runner.add(new sort_bench<sort<2>>{});
        runner.add(new sort_bench<sort<4>>{});
        runner.add(new sort_bench<sort<8>>{});
        runner.add(new sort_bench<sort<12>>{});
        runner.add(new sort_bench<sort<16>>{});
        runner.add(new sort_bench<sort<32>>{});
        runner.add(new sort_bench<sort<48>>{});
        runner.add(new sort_bench<sort<64>>{});
    }

    runner.add(new merge_bench<merge<4>>{});
    // Parameters don't have an effect on cpu sort, so running one sort benchmark is enough
    if (xpu::device::active().backend() != xpu::cpu) {
        runner.add(new merge_bench<merge<8>>{});
        runner.add(new merge_bench<merge<12>>{});
        runner.add(new merge_bench<merge<16>>{});
        runner.add(new merge_bench<merge<32>>{});
        runner.add(new merge_bench<merge<48>>{});
        runner.add(new merge_bench<merge<64>>{});
    }

    runner.run(10);

    return 0;
}
