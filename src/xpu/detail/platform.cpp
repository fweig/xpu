#include "platform.h"

#include <dlfcn.h>
#include <unistd.h>

void *xpu::detail::dlopen_image(const char *name) {
#ifdef __apple__
    return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
#elif defined(__linux__)
    return dlopen(name, RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
#else
    #error "Operating System not supported."
#endif
}

void xpu::detail::get_meminfo(size_t *free, size_t *total) {
#ifdef __apple__
    // macosx is missing _SC_AVPHYS_PAGES...
    // FIXME: any way to do this besides parsing top output???
    *free = 0;
    *total = 0;
#elif defined(__linux__)
    size_t pagesize = sysconf(_SC_PAGESIZE);
    *free = pagesize * sysconf(_SC_AVPHYS_PAGES);
    *total = pagesize * sysconf(_SC_PHYS_PAGES);
#else
    #error "Operating System not supported."
#endif
}
