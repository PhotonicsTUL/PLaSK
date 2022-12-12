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

typedef int (__stdcall* MainProc)(HINSTANCE hInst, LPSTR cmdline);


int WinMain(HINSTANCE hInst, HINSTANCE, LPSTR cmdline, int) {
    setupPythonPath();
    HMODULE plaskgui = LoadLibrary("plaskgui.dll");
    if (!plaskgui) {
        LPSTR buffer;
        auto err = GetLastError();
        size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buffer, 0, NULL);
        std::string err_str(buffer, size);
        LocalFree(buffer);
        std::string message = "Could not start PLaSK GUI: " + err_str;
        if (err == ERROR_MOD_NOT_FOUND)
            message += "\nPlease verify if you have Python " PYTHON_VERSION_STR " installed from https://www.anaconda.com.";
        MessageBox(NULL, message.c_str(), "PLaSK - Error", MB_OK | MB_ICONERROR);
        return 127;
    }
    MainProc plaskgui_main = (MainProc)GetProcAddress(plaskgui, "plaskgui_main");
    return plaskgui_main(hInst, cmdline);
}
