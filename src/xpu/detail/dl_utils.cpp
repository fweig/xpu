#include "dl_utils.h"
#include "log.h"

#include <dlfcn.h>
#include <iostream>

using namespace xpu::detail;

library_loader::library_loader(const std::string &libname) {
    handle = dlopen(libname.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (handle == nullptr) {
        XPU_LOG("Cannot open library '%s': %s", libname.c_str(), dlerror());
    }
    dlerror();
}

library_loader::~library_loader() {
    if (handle) {
        dlclose(handle);
    }
}

void *library_loader::symbol(const std::string &symname) {
    if (not ok()) {
        return nullptr;
    }
    void *sym = dlsym(handle, symname.c_str());
    const char *err = dlerror();
    if (err != nullptr) {
        XPU_LOG("Cannot load symbol '%s': %s", symname.c_str(), err);
        return nullptr;
    }
    return sym;
}

bool library_loader::ok() const {
    return handle != nullptr;
}
