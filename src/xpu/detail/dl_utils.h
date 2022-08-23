#ifndef XPU_DETAIL_DL_UTILS_H
#define XPU_DETAIL_DL_UTILS_H

#include <string>

namespace xpu::detail {

// Some utility classes for loading shared libraries at runtime
class library_loader {

public:
    library_loader(const std::string &);
    ~library_loader();

    void *symbol(const std::string &);

    bool ok() const;

private:
    void *handle = nullptr;

};

template<class T>
class lib_obj {

public:
    using create_f = T*();
    using destroy_f = void(T*);

    T *obj = nullptr;

    lib_obj(const std::string &libname) : lib(libname) {
        if (not lib.ok()) {
            return;
        }
        create = reinterpret_cast<create_f *>(lib.symbol("create"));
        destroy = reinterpret_cast<destroy_f *>(lib.symbol("destroy"));
        obj = create();
    }

    ~lib_obj() {
        if (obj != nullptr) {
            destroy(obj);
        }
    }

    bool ok() const { return lib.ok(); }

private:
    library_loader lib;
    create_f *create = nullptr;
    destroy_f *destroy = nullptr;

};

} // namespace xpu::detail

#endif
