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
#ifndef PLASK_PYTHON_DLL_H
#define PLASK_PYTHON_DLL_H

#include <string>

#include <windows.h>
#include <plask/config.hpp>

#include <iostream>

inline static std::wstring get_anaconda_root() {
    wchar_t* plask_anaconda_env = _wgetenv(L"PLASK_ANACONDA_ROOT");
    if (plask_anaconda_env) return std::wstring(plask_anaconda_env);

    static const WCHAR anaconda_key[] = L"Software\\Python\\ContinuumAnalytics\\Anaconda" PYTHON_VERSION_NUM_STR L"-64\\InstallPath";
    static const WCHAR core_key[] = L"Software\\Python\\PythonCore\\" PYTHON_VERSION_STR L"\\InstallPath";
    WCHAR buffer[MAX_PATH + 1];
    DWORD buf_size = MAX_PATH + 1;

    if (RegGetValueW(HKEY_CURRENT_USER, anaconda_key, NULL, RRF_RT_REG_SZ, NULL, buffer, &buf_size) == ERROR_SUCCESS)
        return std::wstring(buffer);

    if (RegGetValueW(HKEY_LOCAL_MACHINE, anaconda_key, NULL, RRF_RT_REG_SZ, NULL, buffer, &buf_size) == ERROR_SUCCESS)
        return std::wstring(buffer);

    if (RegGetValueW(HKEY_CURRENT_USER, core_key, NULL, RRF_RT_REG_SZ, NULL, buffer, &buf_size) == ERROR_SUCCESS)
        return std::wstring(buffer);

    if (RegGetValueW(HKEY_LOCAL_MACHINE, core_key, NULL, RRF_RT_REG_SZ, NULL, buffer, &buf_size) == ERROR_SUCCESS)
        return std::wstring(buffer);

    // Blind shots
    std::wstring default_user_anaconda = std::wstring(_wgetenv(L"USERPROFILE")) + L"\\AppData\\Local\\Continuum\\anaconda3";
    if (GetFileAttributesW(default_user_anaconda.c_str()) != INVALID_FILE_ATTRIBUTES)
        return default_user_anaconda;

        static const WCHAR* default_system_anaconda = L"C:\\ProgramData\\anaconda3";
    if (GetFileAttributesW(default_system_anaconda) != INVALID_FILE_ATTRIBUTES)
        return default_system_anaconda;
    return std::wstring();
}

void setupPythonPath() {
    std::wstring anaconda_root = get_anaconda_root();
    if (!anaconda_root.empty()) {
        std::wstring path = _wgetenv(L"PATH");
        path = L"PATH="
             + anaconda_root + L";"
             + anaconda_root + L"\\Library\\bin;"
             + anaconda_root + L"\\Library\\usr\\bin;"
             + anaconda_root + L"\\Scripts;"
             + path;
        _wputenv(path.c_str());
        AddDllDirectory(anaconda_root.c_str());
        std::wstring bin_dir = anaconda_root + L"\\Library\\bin";
        AddDllDirectory(bin_dir.c_str());
        bin_dir = anaconda_root + L"\\Library\\usr\\bin";
        AddDllDirectory(bin_dir.c_str());
        std::wstring python_home = L"PYTHONHOME=" + anaconda_root;
        _wputenv(python_home.c_str());
        std::wstring qt_plugins = L"QT_PLUGIN_PATH=" + anaconda_root + L"\\Library\\plugins";
        _wputenv(qt_plugins.c_str());
    }

    // // Add path to plask 'bin' directory to DLL search path
    // wchar_t bin_path_str[MAX_PATH];
    // std::wstring bin_filename(bin_path_str, GetModuleFileNameW(NULL, bin_path_str, MAX_PATH));
    // std::wstring::size_type last_sep = bin_filename.find_last_of(L"\\");
    // std::wstring bin_path = std::wstring::npos ? bin_filename : bin_filename.substr(0, last_sep);
    // AddDllDirectory(bin_path.c_str());
}

#endif
