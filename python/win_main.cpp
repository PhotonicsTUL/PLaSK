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
#include <windows.h>
#include "win_common.hpp"

typedef int (__stdcall* MainProc)(int argc, const wchar_t *argv[]);

int wmain(int argc, const wchar_t *argv[]) {
    setupPythonPath();
    HMODULE plask = LoadLibrary("plaskexe.dll");
    if (!plask) {
        LPSTR buffer;
        auto err = GetLastError();
        size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buffer, 0, NULL);
        std::string err_str(buffer, size);
        LocalFree(buffer);
        std::cerr << "Could not start 'plask': " << err_str;
        if (err == ERROR_MOD_NOT_FOUND)
            std::cerr << "-> Please verify if you have Python " PYTHON_VERSION_STR " installed from https://www.anaconda.com/.\n";
        return 127;
    }
    MainProc plask_main = (MainProc)GetProcAddress(plask, "plask_main");
    return plask_main(argc, argv);
}
