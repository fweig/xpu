#pragma once

#include <string>

// Some utility classes for loading shared libraries at runtime

class LibraryLoader {

public:
    LibraryLoader(const std::string &);
    ~LibraryLoader();

    void *symbol(const std::string &);

private:
    void *handle = nullptr;

};

template<class T>
class LibObj {

public:
    using CreateF = T*();
    using DestroyF = void(T*);

    T *obj = nullptr;

    LibObj(const std::string &libname) : lib(libname) {
        create = reinterpret_cast<CreateF *>(lib.symbol("create"));
        destroy = reinterpret_cast<DestroyF *>(lib.symbol("destroy"));
        obj = create();
    }

    ~LibObj() {
        if (obj != nullptr) {
            destroy(obj);
        }
    }

private:
    LibraryLoader lib;
    CreateF *create = nullptr;
    DestroyF *destroy = nullptr;

};
