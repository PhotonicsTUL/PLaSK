#ifndef PLASK_PYTHON_DLL_H
#define PLASK_PYTHON_DLL_H

#include <string>

#include <windows.h>
#include <plask/config.hpp>

#include <iostream>

void setupPythonPath() {
    static const WCHAR key[] = L"Software\\Python\\PythonCore\\" PYTHON_VERSION_STR L"\\InstallPath";
    WCHAR value[MAX_PATH + 1];
    DWORD buf_size = MAX_PATH + 1;
    LSTATUS result = RegGetValueW(HKEY_CURRENT_USER, key, NULL, RRF_RT_REG_SZ, NULL, value, &buf_size);
    if (result != ERROR_SUCCESS)
        result = RegGetValueW(HKEY_LOCAL_MACHINE, key, NULL, RRF_RT_REG_SZ, NULL, value, &buf_size);
    if (result == ERROR_SUCCESS) {
        std::wstring path = _wgetenv(L"PATH"), python(value);
        path = L"PATH="
             + python + L";"
             + python + L"\\Library\\bin;"
             + python + L"\\Library\\usr\\bin;"
             + python + L"\\Scripts;"
             + path;
        _wputenv(path.c_str());
        std::wstring python_home = L"PYTHONHOME=" + python;
        _wputenv(python_home.c_str());
        std::wstring qt_plugins = L"QT_PLUGIN_PATH=" + python + L"\\Library\\plugins";
        _wputenv(qt_plugins.c_str());
    }
}

#endif
