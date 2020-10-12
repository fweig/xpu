#include XPU_DEVICE_LIBRARY_FRONTEND_H

extern "C" XPU_DEVICE_LIBRARY_NAME *create() {
    return new XPU_DEVICE_LIBRARY_BACKEND_NAME{};
}

extern "C" void destroy(XPU_DEVICE_LIBRARY_NAME *obj) {
    delete obj;
}