/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "loader.hpp"

#include <iostream>
#include "../../exceptions.hpp"

#ifdef PLASK__UTILS_PLUGIN_WINAPI
    #include "../minimal_windows.h"
#else
    #include <dlfcn.h>
#endif

namespace plask {

DynamicLibrary::DynamicLibrary(const std::string& filename, unsigned flags)
: handle(0) {
    open(filename, flags);
}

DynamicLibrary::DynamicLibrary(): handle(0) {}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& to_move) noexcept
    : handle(to_move.handle)
#ifdef PLASK__UTILS_PLUGIN_WINAPI
    , unload(to_move.unload)
#endif
{ to_move.handle = 0; }

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
    //handle = LoadLibraryW(output_buffer->get());
    handle = (handle_t)LoadLibraryA(filename.c_str());
    if (!handle) {
        throw plask::Exception("Could not open dynamic library from file \"{0}\". {1}", filename, GetLastErrorStr());
    }
    unload = !(flags & DONT_CLOSE);
#else
    int mode = RTLD_NOW;
    if (flags & DONT_CLOSE) mode |= RTLD_NODELETE;
    handle = dlopen(filename.c_str(), mode);
    if (!handle) {
        throw plask::Exception("Could not open dynamic library from file \"{0}\". {1}", filename, dlerror());
    }
#endif
}

void DynamicLibrary::close() {
    if (!handle) return;
#ifdef PLASK__UTILS_PLUGIN_WINAPI
    if (unload) {
        if (!FreeLibrary((HINSTANCE)handle))
            throw plask::Exception("Can't close dynamic library: {0}", GetLastErrorStr());
    }
#else
    if (dlclose(handle))
        throw plask::Exception("Can't close dynamic library: {0}", dlerror());
#endif
    handle = 0;
}

void * DynamicLibrary::getSymbol(const std::string &symbol_name) const {
    if (!handle)
        throw plask::Exception("Trying to get symbol from dynamic library which is not opened.");

    return
#ifdef PLASK__UTILS_PLUGIN_WINAPI
        (void*) GetProcAddress((HINSTANCE)handle, symbol_name.c_str());
#else
        dlsym(handle, symbol_name.c_str());
#endif
}

void *DynamicLibrary::requireSymbol(const std::string &symbol_name) const {
    void* result = getSymbol(symbol_name);
    if (!result)
        throw plask::Exception("There is no symbol \"{0}\" in dynamic library.", symbol_name);
    return result;
}

DynamicLibrary::handle_t DynamicLibrary::release() {
     handle_t r = handle;
     handle = 0;
     return r;
}

constexpr const char* DynamicLibrary::DEFAULT_EXTENSION;


}   // namespace plask
