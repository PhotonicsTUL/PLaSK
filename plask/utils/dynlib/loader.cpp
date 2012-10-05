#include "loader.h"

#include <iostream>
#include "../../exceptions.h"

namespace plask {

DynamicLibrary::DynamicLibrary(const std::string& filename, unsigned flags)
: handler(0) {
    open(filename, flags);
}

DynamicLibrary::DynamicLibrary(): handler(0) {}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& to_move)
    : handler(to_move.handler) { to_move.handler = 0; }

DynamicLibrary::~DynamicLibrary() {
    close();
}

void DynamicLibrary::open(const std::string &filename, unsigned flags) {
    close();    // close if something is already open
#ifdef PLASK__UTILS_PLUGIN_WINAPI
    //const int length = MultiByteToWideChar(CP_UTF8, 0, filename.data(), filename.size(), 0, 0);
    //std::unique_ptr<wchar_t> output_buffer(new wchar_t [length]);
    //MultiByteToWideChar(CP_UTF8, 0, filename.data(), filename.size(), output_buffer.get(), length);
    //handler = LoadLibraryW(output_buffer->get());
    handler = LoadLibraryA(filename.c_str());
    if (!handler) {
        throw plask::Exception("Could not open dynamic library from file \"%1%\".", filename);
    }
    unload = !(flags & DONT_CLOSE);
#else
    int mode = RTLD_NOW;
    if (flags & DONT_CLOSE) mode |= RTLD_NODELETE;
    handler = dlopen(filename.c_str(), mode);
    if (!handler) {
        throw plask::Exception("Could not open dynamic library from %1%", dlerror());
    }
#endif
}

void DynamicLibrary::close() {
    if (!handler) return;
#ifdef PLASK__UTILS_PLUGIN_WINAPI
    if (unload) {
        if (!FreeLibrary(handler))
            throw plask::Exception("Can't close dynamic library: %1%", (const char*)GetLastError());
    }
#else
    if (dlclose(handler))
        throw plask::Exception("Can't close dynamic library: %1%", dlerror());
#endif
    handler = 0;
}

void * DynamicLibrary::getSymbol(const std::string &symbol_name) const {
    if (!handler)
        throw plask::Exception("Trying to get symbol from dynamic library which is not opened.");

    return
#ifdef PLASK__UTILS_PLUGIN_WINAPI
        (void*) GetProcAddress(handler, symbol_name.c_str());
#else
        dlsym(handler, symbol_name.c_str());
#endif
}

void *DynamicLibrary::requireSymbol(const std::string &symbol_name) const {
    void* result = getSymbol(symbol_name);
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

