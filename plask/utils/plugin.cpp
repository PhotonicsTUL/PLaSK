#include "plugin.h"

#include <iostream>
#include "../exceptions.h"

namespace plask {

DynamicLibrary::DynamicLibrary(const std::string &filename)
: handler(0) {
    open(filename);
}

DynamicLibrary::DynamicLibrary(): handler(0) {}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& to_move)
    : handler(to_move.handler) { to_move.handler = 0; }

DynamicLibrary::~DynamicLibrary() {
    close();
}

void DynamicLibrary::open(const std::string &filename) {
    close();    // close if something is already open
#ifdef PLASK__UTILS_PLUGIN_WINAPI
    handler = LoadLibrary(filename.c_str());
#else
    handler = dlopen(filename.c_str(), RTLD_NOW);
#endif
    if (!handler)
        throw plask::Exception("Could not open dynamic library from file \"%1%\".", filename);
}

void DynamicLibrary::close() {
    if (!handler) return;
#ifdef PLASK__UTILS_PLUGIN_WINAPI
    if (!FreeLibrary(handler))
        throw plask::Exception("Can't close dynamic library: %1%", (const char*)GetLastError());
#else
    if (dlclose(handler))
        throw plask::Exception("Can't close dynamic library: %1%", dlerror());
#endif
    handler = 0;
}

void * DynamicLibrary::operator [](const std::string &symbol_name) {
    if (!handler)
        throw plask::Exception("Trying to get symbol from dynamic library which is not opened.");

    return
#if PLASK__UTILS_PLUGIN_WINAPI
        (void*) GetProcAddress(handler, symbol_name.c_str());
#else
        dlsym(handler, symbol_name.c_str());
#endif
}

void *DynamicLibrary::requireSymbol(const std::string &symbol_name) {
    void* result = operator[](symbol_name);
    if (!result)
        throw plask::Exception("There is no symbol \"%1%\" in dynamic library.", symbol_name);
    return result;
}

DynamicLibrary::handler_t DynamicLibrary::release() {
     handler_t r = handler;
     handler = 0;
     return r;
}


}   // namespace plask

