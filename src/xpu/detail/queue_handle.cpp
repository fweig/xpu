#include "common.h"
#include "runtime.h"

using namespace xpu::detail;

queue_handle::queue_handle() : queue_handle(runtime::instance().active_device()) {
}

queue_handle::queue_handle(device dev) : dev(dev) {
    handle = runtime::instance().create_queue(dev);
}

queue_handle::~queue_handle() {
    runtime::instance().destroy_queue(*this);
}
