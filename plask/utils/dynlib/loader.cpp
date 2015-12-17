#include "loader.h"

#include <iostream>
#include "../../exceptions.h"

namespace plask {

DynamicLibrary::DynamicLibrary(const std::string& filename, unsigned flags)
: handler(0) {
    open(filename, flags);
}

DynamicLibrary::DynamicLibrary(): handler(0) {}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& to_move) noexcept
    : handler(to_move.handler)
#ifdef PLASK__UTILS_PLUGIN_WINAPI
    , unload(to_move.unload)
#endif    
{ to_move.handler = 0; }

DynamicLibrary::~DynamicLibrary() {
    close();
}

#ifdef PLASK__UTILS_PLUGIN_WINAPI
// Create a string with last error message, copied from http://www.codeproject.com/Tips/479880/GetLastError-as-std-string
std::string GetLastErrorStr()
{
  DWORD error = GetLastError();
  if (error)
  {
    LPVOID lpMsgBuf;
    DWORD bufLen = FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        error,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR) &lpMsgBuf,
        0, NULL );
    if (bufLen)
    {
      LPCSTR lpMsgStr = (LPCSTR)lpMsgBuf;
      std::string result(lpMsgStr, lpMsgStr+bufLen);

      LocalFree(lpMsgBuf);

      return result;
    }
  }
  return std::string();
}
#endif

void DynamicLibrary::open(const std::string &filename, unsigned flags) {
    close();    // close if something is already opened
#ifdef PLASK__UTILS_PLUGIN_WINAPI
    //const int length = MultiByteToWideChar(CP_UTF8, 0, filename.data(), filename.size(), 0, 0);
    //std::unique_ptr<wchar_t> output_buffer(new wchar_t [length]);
    //MultiByteToWideChar(CP_UTF8, 0, filename.data(), filename.size(), output_buffer.get(), length);
    //handler = LoadLibraryW(output_buffer->get());
    handler = LoadLibraryA(filename.c_str());
    if (!handler) {
        throw plask::Exception("Could not open dynamic library from file \"{0}\". {1}", filename, GetLastErrorStr());
    }
    unload = !(flags & DONT_CLOSE);
#else
    int mode = RTLD_NOW;
    if (flags & DONT_CLOSE) mode |= RTLD_NODELETE;
    handler = dlopen(filename.c_str(), mode);
    if (!handler) {
        throw plask::Exception("Could not open dynamic library from file \"{0}\". {1}", filename, dlerror());
    }
#endif
}

void DynamicLibrary::close() {
    if (!handler) return;
#ifdef PLASK__UTILS_PLUGIN_WINAPI
    if (unload) {
        if (!FreeLibrary(handler))
            throw plask::Exception("Can't close dynamic library: {0}", GetLastErrorStr());
    }
#else
    if (dlclose(handler))
        throw plask::Exception("Can't close dynamic library: {0}", dlerror());
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
        throw plask::Exception("There is no symbol \"{0}\" in dynamic library.", symbol_name);
    return result;
}

DynamicLibrary::handler_t DynamicLibrary::release() {
     handler_t r = handler;
     handler = 0;
     return r;
}


}   // namespace plask

