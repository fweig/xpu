#include "../host.h"

#include "../detail/exceptions.h"
#include "../detail/runtime.h"
#include "../detail/timers.h"
#include "../detail/type_info.h"

#include "queue.tpp"

void xpu::initialize(settings settings) {
    detail::runtime::instance().initialize(settings);
}

template<typename I>
inline void xpu::preload() {
    static_assert(detail::is_device_image_v<I>, "Invalid image");
    detail::runtime::instance().preload_image<I>();
}

inline void *xpu::malloc_pinned(size_t bytes) {
    return detail::runtime::instance().malloc_host(bytes);
}

template<typename T>
T *xpu::malloc_pinned(size_t count) {
    return static_cast<T *>(malloc_pinned(count * sizeof(T)));
}

inline void *xpu::malloc_device(size_t bytes) {
    return detail::runtime::instance().malloc_device(bytes);
}

template<typename T>
T *xpu::malloc_device(size_t count) {
    return static_cast<T *>(malloc_device(count * sizeof(T)));
}

inline void *xpu::malloc_managed(size_t bytes) {
    return detail::runtime::instance().malloc_shared(bytes);
}

template<typename T>
T *xpu::malloc_managed(size_t count) {
    return static_cast<T *>(malloc_managed(count * sizeof(T)));
}

void xpu::free(void *ptr) {
    detail::runtime::instance().free(ptr);
}

inline void xpu::stack_alloc(size_t bytes) {
    detail::buffer_registry::instance().stack_alloc(
        detail::runtime::instance().active_device(),
        bytes
    );
}

inline void xpu::stack_pop(void *head) {
    detail::buffer_registry::instance().stack_pop(
        detail::runtime::instance().active_device(),
        head
    );
}

inline std::vector<xpu::device> xpu::device::all() {
    auto dev_impl = detail::runtime::instance().get_devices();

    std::vector<xpu::device> devices;
    devices.reserve(dev_impl.size());

    for (auto &d : dev_impl) {
        devices.push_back(device{d});
    }

    return devices;
}

inline xpu::device xpu::device::active() {
    return device{detail::runtime::instance().active_device()};
}

inline xpu::device::device() {
    m_impl = detail::runtime::instance().get_devices()[0];
}

inline xpu::device::device(std::string_view xpuid) {
    m_impl = detail::runtime::instance().get_device(xpuid);
}

inline xpu::device::device(driver_t backend, int device_) {
    m_impl = detail::runtime::instance().get_device(static_cast<detail::driver_t>(backend), device_);
}

inline xpu::device::device(int id) {
    m_impl = detail::runtime::instance().get_device(id);
}

inline xpu::device_prop::device_prop(xpu::device dev) {
    m_prop = detail::runtime::instance().device_properties(dev.id());
}

template<typename Kernel>
const char *xpu::get_name() {
    return detail::type_name<Kernel>();
}

template<typename Func, typename... Args>
void xpu::call(Args&&... args) {
    static_assert(detail::is_function_v<Func>, "Invalid function");
    detail::runtime::instance().call<Func>(std::forward<Args>(args)...);
}

template<typename C>
void xpu::set(const typename C::data_t &symbol) {
    static_assert(detail::is_constant_v<C>, "Invalid constant");
    detail::runtime::instance().set_constant<C>(symbol);
}

inline xpu::ptr_prop::ptr_prop(const void *ptr) {
    detail::runtime::instance().get_ptr_prop(ptr, &m_prop);
}

template<typename T>
xpu::buffer_prop::buffer_prop(const buffer<T> &buf) {
    detail::buffer_data entry = detail::buffer_registry::instance().get(buf.get());
    m_size_bytes = entry.size;
    m_size = entry.size / sizeof(T);
    m_device = entry.ptr;
    m_host = entry.host_ptr;
    m_type = static_cast<xpu::buffer_type>(entry.type);
}

template<typename T>
xpu::h_view<T>::h_view(buffer<T> &buf) {
    detail::buffer_data entry = detail::buffer_registry::instance().get(buf.get());
    m_data = static_cast<T *>(entry.host_ptr);
    m_size = entry.size / sizeof(T);
}

template<typename T>
T &xpu::h_view<T>::operator[](size_t i) {
    XPU_CHECK_RANGE("h_view::operator[]", i, m_size);
    return m_data[i];
}

template<typename T>
const T &xpu::h_view<T>::operator[](size_t i) const {
    XPU_CHECK_RANGE("h_view::operator[]", i, m_size);
    return m_data[i];
}

template<typename T>
T &xpu::h_view<T>::at(size_t i) {
    XPU_CHECK_RANGE("h_view::at", i, m_size);
    return m_data[i];
}

template<typename T>
const T &xpu::h_view<T>::at(size_t i) const {
    XPU_CHECK_RANGE("h_view::at", i, m_size);
    return m_data[i];
}

inline void xpu::push_timer(std::string_view name) {
    detail::push_timer(name);
}

inline xpu::timings xpu::pop_timer() {
    return timings{detail::pop_timer()};
}

inline xpu::scoped_timer::scoped_timer(std::string_view name, xpu::timings *t) : m_t(t) {
    detail::push_timer(name);
}

inline xpu::scoped_timer::~scoped_timer() {
    if (m_t == nullptr) {
        detail::pop_timer();
    } else {
        *m_t = timings{detail::pop_timer()};
    }
}

inline void xpu::t_add_bytes(size_t bytes) {
    detail::add_bytes_timer(bytes);
}

template<typename Kernel>
inline void xpu::k_add_bytes(size_t bytes) {
    detail::add_bytes_kernel(detail::type_name<Kernel>(), bytes);
}

namespace xpu::detail {
inline double bytes_per_ms_to_gb_per_sec(size_t bytes, double ms) {
    return bytes / (ms * 1e6);
}
}

inline double xpu::kernel_timings::throughput() const {
    return detail::bytes_per_ms_to_gb_per_sec(m_t.bytes_input, total());
}

inline double xpu::timings::throughput() const {
    return detail::bytes_per_ms_to_gb_per_sec(m_t.bytes_input, wall());
}

inline double xpu::timings::throughput_kernels() const {
    return detail::bytes_per_ms_to_gb_per_sec(m_t.bytes_input, kernel_time());
}

inline double xpu::timings::throughput_copy(xpu::direction dir) const {
    switch (dir) {
    case h2d:
        return detail::bytes_per_ms_to_gb_per_sec(m_t.bytes_h2d, m_t.copy_h2d);
    case d2h:
        return detail::bytes_per_ms_to_gb_per_sec(m_t.bytes_d2h, m_t.copy_d2h);
    }

    throw std::runtime_error("invalid direction"); // unreachable
}

inline double xpu::timings::throughput_memset() const {
    return detail::bytes_per_ms_to_gb_per_sec(m_t.bytes_memset, m_t.memset);
}

inline std::vector<xpu::kernel_timings> xpu::timings::kernels() const {
    std::vector<kernel_timings> kernels;
    kernels.reserve(m_t.kernels.size());
    for (auto &k : m_t.kernels) {
        kernels.emplace_back(k);
    }
    return kernels;
}

inline std::vector<xpu::timings> xpu::timings::children() const {
    std::vector<timings> children;
    children.reserve(m_t.children.size());
    for (auto &c : m_t.children) {
        children.emplace_back(c);
    }
    return children;
}

inline xpu::kernel_timings xpu::timings::kernel(std::string_view name) const {
    auto it = std::find_if(m_t.kernels.begin(), m_t.kernels.end(), [&](const auto &k) {
        return k.name == name;
    });
    if (it == m_t.kernels.end()) {
        detail::kernel_timings kt;
        kt.name = name;
        return kernel_timings{kt};
    }
    return kernel_timings{*it};
}
