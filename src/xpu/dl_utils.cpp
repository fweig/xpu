#include "dl_utils.h"

#include <dlfcn.h>
#include <iostream>

LibraryLoader::LibraryLoader(const std::string &libname) {
    handle = dlopen(libname.c_str(), RTLD_LAZY);
    if (handle == nullptr) {
        std::cout << "Cannot open library 'libname': " << dlerror() << std::endl;
    }
    dlerror();
}

LibraryLoader::~LibraryLoader() {
    if (handle) {
        dlclose(handle);
    }
}

void *LibraryLoader::symbol(const std::string &symname) {
    void *sym = dlsym(handle, symname.c_str());
    const char *err = dlerror();
    if (err != nullptr) {
        std::cout << "Cannot load symbol '" << symname << "': " << err << std::endl;
        return nullptr;
    }
    return sym;
}