#include "common.h"
#include "backend.h"
#include "runtime.h"

using namespace xpu::detail;

queue_handle::queue_handle() : queue_handle(runtime::instance().active_device()) {
}

queue_handle::queue_handle(device dev) : dev(dev) {
    backend::call(dev.backend, &backend_base::create_queue, &handle, dev.device_nr);
}

queue_handle::~queue_handle() {
    backend::call(dev.backend, &backend_base::destroy_queue, handle);
}
