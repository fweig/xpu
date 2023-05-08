#ifndef XPU_DRIVER_SYCL_DEVICE_H
#define XPU_DRIVER_SYCL_DEVICE_H

#include "../../backend.h"
#include "../../constant_memory.h"
#include "../../parallel_merge.h"
#include "cmem_impl.h"
#include "sycl_driver.h"

#include <sycl/sycl.hpp>

#define XPU_DETAIL_ASSERT(x) assert(x)

// Pull printf into global namespace, to be consistent with other backends.
using sycl::ext::oneapi::experimental::printf;

template<typename T>
struct sycl::is_device_copyable<xpu::buffer<T>> : std::true_type {};

int xpu::abs(int x) { return sycl::abs(x); }
float xpu::abs(float x) { return sycl::fabs(x); }
float xpu::acos(float x) { return sycl::acos(x); }
float xpu::acosh(float x) { return sycl::acosh(x); }
float xpu::acospi(float x) { return sycl::acospi(x); }
float xpu::asin(float x) { return sycl::asin(x); }
float xpu::asinh(float x) { return sycl::asinh(x); }
float xpu::asinpi(float x) { return sycl::asinpi(x); }
float xpu::atan(float x) { return sycl::atan(x); }
float xpu::atan2(float y, float x) { return sycl::atan2(y, x); }
float xpu::atan2pi(float y, float x) { return sycl::atan2pi(y, x); }
float xpu::atanh(float x) { return sycl::atanh(x); }
float xpu::atanpi(float x) { return sycl::atanpi(x); }
float xpu::cbrt(float x) { return sycl::cbrt(x); }
float xpu::ceil(float x) { return sycl::ceil(x); }
float xpu::copysign(float x, float y) { return sycl::copysign(x, y); }
float xpu::cos(float x) { return sycl::cos(x); }
float xpu::cosh(float x) { return sycl::cosh(x); }
float xpu::cospi(float x) { return sycl::cospi(x); }
float xpu::erf(float x) { return sycl::erf(x); }
float xpu::erfc(float x) { return sycl::erfc(x); }
float xpu::exp(float x) { return sycl::exp(x); }
float xpu::exp2(float x) { return sycl::exp2(x); }
float xpu::exp10(float x) { return sycl::exp10(x); }
float xpu::expm1(float x) { return sycl::expm1(x); }
float xpu::fdim(float x, float y) { return sycl::fdim(x, y); }
float xpu::floor(float x) { return sycl::floor(x); }
float xpu::fma(float x, float y, float z) { return sycl::fma(x, y, z); }
float xpu::fmod(float x, float y) { return sycl::fmod(x, y); }
float xpu::hypot(float x, float y) { return sycl::hypot(x, y); }
int xpu::ilogb(float x) { return sycl::ilogb(x); }
bool xpu::isfinite(float x) { return sycl::isfinite(x); }
bool xpu::isinf(float x) { return sycl::isinf(x); }
bool xpu::isnan(float x) { return sycl::isnan(x); }
float xpu::ldexp(float x, int exp) { return sycl::ldexp(x, exp); }
long long int xpu::llrint(float x) { return sycl::rint(x); }
long long int xpu::llround(float x) { return sycl::round(x); }
float xpu::log(float x) { return sycl::log(x); }
float xpu::log10(float x) { return sycl::log10(x); }
float xpu::log1p(float x) { return sycl::log1p(x); }
float xpu::log2(float x) { return sycl::log2(x); }
float xpu::logb(float x) { return sycl::logb(x); }
long int xpu::lrint(float x) { return sycl::rint(x); }
long int xpu::lround(float x) { return sycl::round(x); }
int xpu::max(int x, int y) { return sycl::max(x, y); }
unsigned int xpu::max(unsigned int x, unsigned int y) { return sycl::max(x, y); }
long long int xpu::max(long long int x, long long int y) { return sycl::max(x, y); }
unsigned long long int xpu::max(unsigned long long int x, unsigned long long int y) { return sycl::max(x, y); }
float xpu::max(float x, float y) { return sycl::fmax(x, y); }
int xpu::min(int x, int y) { return sycl::min(x, y); }
unsigned int xpu::min(unsigned int x, unsigned int y) { return sycl::min(x, y); }
long long int xpu::min(long long int x, long long int y) { return sycl::min(x, y); }
unsigned long long int xpu::min(unsigned long long int x, unsigned long long int y) { return sycl::min(x, y); }
float xpu::min(float x, float y) { return sycl::fmin(x, y); }
float xpu::nan(const char* /*tagp*/) { return sycl::nan(1u); }
float xpu::norm3d(float x, float y, float z) { return sycl::length(sycl::vec<float, 3>{x, y, z}); }
float xpu::norm4d(float x, float y, float z, float w) { return sycl::length(sycl::vec<float, 4>{x, y, z, w}); }
float xpu::pow(float x, float y) { return sycl::pow(x, y); }
float xpu::rcbrt(float x) { return 1.f / cbrt(x); }
float xpu::remainder(float x, float y) { return sycl::remainder(x, y); }
float xpu::remquo(float x, float y, int* quo) { return sycl::remquo(x, y, quo); }
float xpu::rint(float x) { return sycl::rint(x); }
float xpu::rhypot(float x, float y) { return 1.f / hypot(x, y); }
float xpu::rnorm3d(float x, float y, float z) { return 1.f / norm3d(x, y, z); }
float xpu::rnorm4d(float x, float y, float z, float w) { return 1.f / norm4d(x, y, z, w); }
float xpu::round(float x) { return sycl::round(x); }
float xpu::rsqrt(float x) { return sycl::rsqrt(x); }
bool xpu::signbit(float x) { return sycl::signbit(x); }
void xpu::sincos(float x, float* sptr, float* cptr) { *sptr = sycl::sincos(x, cptr); }
void xpu::sincospi(float x, float* sptr, float* cptr) { *sptr = sycl::sincos(pi() * x, cptr); }
float xpu::sin(float x) { return sycl::sin(x); }
float xpu::sinh(float x) { return sycl::sinh(x); }
float xpu::sinpi(float x) { return sycl::sinpi(x); }
float xpu::sqrt(float x) { return sycl::sqrt(x); }
float xpu::tan(float x) { return sycl::tan(x); }
float xpu::tanh(float x) { return sycl::tanh(x); }
float xpu::tanpi(float x) { return sycl::tanpi(x); }
float xpu::tgamma(float x) { return sycl::tgamma(x); }
float xpu::trunc(float x) { return sycl::trunc(x); }

namespace xpu::detail {
template<typename T>
using atomic_ref = sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device>;
} // namespace xpu::detail

int xpu::atomic_cas(int *addr, int compare, int val) {
    return detail::atomic_ref<int>{*addr}.compare_exchange_strong(compare, val);
}

unsigned int xpu::atomic_cas(unsigned int *addr, unsigned int compare, unsigned int val) {
    return detail::atomic_ref<unsigned int>{*addr}.compare_exchange_strong(compare, val);
}

int xpu::atomic_cas_block(int *addr, int compare, int val) { return atomic_cas(addr, compare, val); }
unsigned int xpu::atomic_cas_block(unsigned int *addr, unsigned int compare, unsigned int val) { return atomic_cas(addr, compare, val); }

int xpu::atomic_add(int *addr, int val) { return detail::atomic_ref<int>{*addr}.fetch_add(val); }
unsigned int xpu::atomic_add(unsigned int *addr, unsigned int val) { return detail::atomic_ref<unsigned int>{*addr}.fetch_add(val); }
int xpu::atomic_add_block(int *addr, int val) { return atomic_add(addr, val); }
unsigned int xpu::atomic_add_block(unsigned int *addr, unsigned int val) { return atomic_add(addr, val); }

int xpu::atomic_sub(int *addr, int val) { return detail::atomic_ref<int>{*addr}.fetch_sub(val); }
unsigned int xpu::atomic_sub(unsigned int *addr, unsigned int val) { return detail::atomic_ref<unsigned int>{*addr}.fetch_sub(val); }
int xpu::atomic_sub_block(int *addr, int val) { return atomic_sub(addr, val); }
unsigned int xpu::atomic_sub_block(unsigned int *addr, unsigned int val) { return atomic_sub(addr, val); }

int xpu::atomic_and(int *addr, int val) { return detail::atomic_ref<int>{*addr}.fetch_and(val); }
unsigned int xpu::atomic_and(unsigned int *addr, unsigned int val) { return detail::atomic_ref<unsigned int>{*addr}.fetch_and(val); }
int xpu::atomic_and_block(int *addr, int val) { return atomic_and(addr, val); }
unsigned int xpu::atomic_and_block(unsigned int *addr, unsigned int val) { return atomic_and(addr, val); }

int xpu::atomic_or(int *addr, int val) { return detail::atomic_ref<int>{*addr}.fetch_or(val); }
unsigned int xpu::atomic_or(unsigned int *addr, unsigned int val) { return detail::atomic_ref<unsigned int>{*addr}.fetch_or(val); }
int xpu::atomic_or_block(int *addr, int val) { return atomic_or(addr, val); }
unsigned int xpu::atomic_or_block(unsigned int *addr, unsigned int val) { return atomic_or(addr, val); }

int xpu::atomic_xor(int *addr, int val) { return detail::atomic_ref<int>{*addr}.fetch_xor(val); }
unsigned int xpu::atomic_xor(unsigned int *addr, unsigned int val) { return detail::atomic_ref<unsigned int>{*addr}.fetch_xor(val); }
int xpu::atomic_xor_block(int *addr, int val) { return atomic_xor(addr, val); }
unsigned int xpu::atomic_xor_block(unsigned int *addr, unsigned int val) { return atomic_xor(addr, val); }

int xpu::float_as_int(float x) { return sycl::bit_cast<int>(x); }
float xpu::int_as_float(int x) { return sycl::bit_cast<float>(x); }

void xpu::barrier(tpos &pos) {
    detail::tpos_impl &impl = pos.impl(detail::internal_fn);
    sycl::group_barrier(impl.group());
}

template<typename T, int BlockSize>
class xpu::block_scan<T, BlockSize, xpu::sycl> {

public:
    struct storage_t {};

    template <typename ContextT>
    XPU_D block_scan(ContextT &ctx, storage_t &) : m_pos(ctx.pos()) {}

    XPU_D block_scan(tpos &pos, storage_t &) : m_pos(pos) {}

    XPU_D void exclusive_sum(T input, T &output) {
        detail::tpos_impl &impl = m_pos.impl(detail::internal_fn);
        output = sycl::exclusive_scan_over_group(impl.group(), input, sycl::plus<T>{});
    }

    template<typename ScanOp>
    XPU_D void exclusive_sum(T input, T &output, T initial_value, ScanOp op) {
        detail::tpos_impl &impl = m_pos.impl(detail::internal_fn);
        output = sycl::exclusive_scan_over_group(impl.group(), input, initial_value, op);
    }

    XPU_D void inclusive_sum(T input, T &output) {
        detail::tpos_impl &impl = m_pos.impl(detail::internal_fn);
        output = sycl::inclusive_scan_over_group(impl.group(), input, sycl::plus<T>{});
    }

    template<typename ScanOp>
    XPU_D void inclusive_sum(T input, T &output, T initial_value, ScanOp op) {
        detail::tpos_impl &impl = m_pos.impl(detail::internal_fn);
        output = sycl::inclusive_scan_over_group(impl.group(), input, initial_value, op);
    }

private:
    tpos &m_pos;

};

template<typename T, int BlockSize>
class xpu::block_reduce<T, BlockSize, xpu::sycl> {

public:
    struct storage_t {};

    template <typename ContextT>
    XPU_D block_reduce(ContextT &ctx, storage_t &) : m_pos(ctx.pos()) {}

    XPU_D block_reduce(tpos &pos, storage_t &) : m_pos(pos) {}

    XPU_D T sum(T input) {
        detail::tpos_impl &impl = m_pos.impl(detail::internal_fn);
        return sycl::reduce_over_group(impl.group(), input, sycl::plus<T>{});
    }

// Disable on SYCL until I figure out how to add a custom reduction operator.
#if !XPU_IS_SYCL
    template<typename ReduceOp>
    XPU_D T reduce(T input, ReduceOp op) {
        detail::tpos_impl &impl = m_pos.impl(detail::internal_fn);
        return sycl::reduce_over_group(impl.group(), input, op);
    }
#endif

private:
    tpos &m_pos;

};

template<typename Key, typename T, int BlockSize, int ItemsPerThread>
class xpu::block_sort<Key, T, BlockSize, ItemsPerThread, xpu::sycl> {

public:
    using key_t = Key;
    using data_t = T;
    using block_merge_t = block_merge<data_t, BlockSize, ItemsPerThread>;

    static_assert(std::is_trivially_constructible_v<data_t>, "Sorted type needs trivial constructor.");

    using storage_t = typename block_merge_t::storage_t;

    block_sort(tpos &pos, storage_t &storage) : m_pos(pos), m_storage(storage) {}

    template<typename KeyGetter>
    data_t *sort(data_t *data, size_t N, data_t *buf, KeyGetter &&getKey) {
        constexpr int ItemsPerBlock = BlockSize * ItemsPerThread;

        size_t nItemBlocks = (N + ItemsPerBlock - 1) / ItemsPerBlock;
        data_t local_data[ItemsPerThread];

        for (size_t i = 0; i < nItemBlocks; i++) {
            size_t start = i * ItemsPerBlock;
            int n_items = 0;

            xpu::barrier(m_pos);
            for (size_t b = 0; b < ItemsPerThread; b++) {
                // TODO: poor accesses pattern, try warp shuffle
                size_t thread_idx = m_pos.thread_idx_x() * ItemsPerThread + b;
                size_t from = start + thread_idx;
                if (from < N) {
                    local_data[b] = data[from];
                    n_items++;
                }
                xpu::barrier(m_pos);
            }

            xpu::barrier(m_pos);

            selection_sort_thread(local_data, n_items, getKey);

            xpu::barrier(m_pos);

            for (size_t b = 0; b < ItemsPerThread; b++) {
                size_t to = start + m_pos.thread_idx_x() * ItemsPerThread + b;
                if (to < N) {
                    data[to] = local_data[b];
                }
            }

            xpu::barrier(m_pos);
        }

        xpu::barrier(m_pos);

        data_t *src = data;
        data_t *dst = buf;
        block_merge_t block_merge{m_pos, m_storage};

        for (size_t blockSize = ItemsPerThread; blockSize < N; blockSize *= 2) {
        // for (size_t blockSize = ItemsPerThread; blockSize < ItemsPerThread+1; blockSize *= 2) {

            size_t carryStart = 0;
            for (size_t st = 0; st + blockSize < N; st += 2 * blockSize) {
                size_t st2 = st + blockSize;
                size_t blockSize2 = min((unsigned long long int)(N - st2), (unsigned long long int)blockSize);
                carryStart = st2 + blockSize2;

                auto comp = [&](const data_t &a, const data_t &b) { return getKey(a) < getKey(b); };
                block_merge.merge(&src[st], blockSize, &src[st2], blockSize2, &dst[st], comp);

                xpu::barrier(m_pos);
            }

            for (size_t i = carryStart + m_pos.thread_idx_x(); i < N; i += m_pos.block_dim_x()) {
                dst[i] = src[i];
            }

            xpu::barrier(m_pos);

            std::swap(src, dst);
        }

        return src;
    }

private:
    tpos &m_pos;
    storage_t &m_storage;

    template<typename KeyGetter>
    void selection_sort_thread(data_t *local_data, int N, KeyGetter &&get_key) {
        for (int i = 0; i < N; ++i) {
            int min_idx = i;
            for (int j = i + 1; j < N; ++j) {
                if (get_key(local_data[j]) < get_key(local_data[min_idx])) {
                    min_idx = j;
                }
            }
            std::swap(local_data[i], local_data[min_idx]);
        }
    }

};

template<typename Key, int BlockSize, int ItemsPerThread>
class xpu::block_merge<Key, BlockSize, ItemsPerThread, xpu::sycl> {

public:
    using data_t = Key;
    using impl_t = detail::parallel_merge<Key, BlockSize, ItemsPerThread>;

    static_assert(std::is_trivially_constructible_v<data_t>, "Merged type needs trivial constructor.");

    using storage_t = typename impl_t::storage_t;

    XPU_D block_merge(tpos& pos, storage_t &storage) : impl(pos, storage) {}

    template<typename Compare>
    XPU_D void merge(const data_t *a, size_t size_a, const data_t *b, size_t size_b, data_t *dst, Compare &&comp) {
        impl.merge(a, size_a, b, size_b, dst, comp);
    }

private:
    impl_t impl;

};

template<typename F>
struct xpu::detail::action_runner<xpu::detail::constant_tag, F> {
    using data_t = typename F::data_t;
    static int call(const data_t &val) {
        constant_memory<F> = val;
        return 0;
    }
};

template<typename K, typename... Args>
struct xpu::detail::action_runner<xpu::detail::kernel_tag, K, void(K::*)(xpu::kernel_context<typename K::shared_memory, typename K::constants> &, Args...)> {

    using shared_memory = typename K::shared_memory;
    using constants = typename K::constants;
    using context = kernel_context<shared_memory, constants>;

    static int call(kernel_launch_info launch_info, Args... args) {
        dim block_dim = K::block_size::value;
        dim grid_dim{};
        launch_info.g.get_compute_grid(block_dim, grid_dim);

        XPU_LOG("Calling kernel '%s' [block_dim = (%d, %d, %d), grid_dim = (%d, %d, %d)] with SYCL driver.", type_name<K>(), block_dim.x, block_dim.y, block_dim.z, grid_dim.x, grid_dim.y, grid_dim.z);

        auto *driver = static_cast<sycl_driver *>(backend::get(sycl));
        sycl::queue queue = driver->get_queue(launch_info.queue_handle);

        sycl::range<3> global_range{size_t(grid_dim.x), size_t(grid_dim.y), size_t(grid_dim.z)};
        sycl::range<3> local_range{size_t(block_dim.x), size_t(block_dim.y), size_t(block_dim.z)};
        cmem_traits<constants> cmem_traits{};
        auto cmem_buffers = cmem_traits.make_buffers();

        global_range = global_range * local_range;

        sycl::event ev = queue.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<shared_memory, 0> shared_memory_acc{cgh};
            auto cmem_accessors = cmem_traits.make_accessors(cmem_buffers, cgh);
            constants cmem{internal_ctor, cmem_accessors};

            sycl::stream out{0, 0, cgh};
            cgh.parallel_for<K>(sycl::nd_range<3>{global_range, local_range}, [=](sycl::nd_item<3> item) {
                // WTF: icpx sometimes optimizes out the kernel call (when using O2)
                // if we dont add the print statement
                if (item.get_global_id(0) == static_cast<size_t>(-1)) {
                    out << "";
                }
                shared_memory &smem = shared_memory_acc;
                tpos pos{internal_ctor, item};
                context ctx{internal_ctor, pos, smem, cmem};
                K{}(ctx, args...);
            });
        });

        if (launch_info.ms != nullptr) {
            ev.wait();
            int64_t nanoseconds = ev.get_profiling_info<sycl::info::event_profiling::command_end>() - ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            *launch_info.ms = float(nanoseconds) / 1e6f;
        } else if (launch_info.queue_handle != nullptr) {
            ev.wait();
        }

        return 0;
    }

};

#endif
