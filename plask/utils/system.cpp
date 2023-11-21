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
#include "system.hpp"
#include <cstdlib>

#if defined(_MSC_VER) || defined(__MINGW32__)
#   define PLASK_SYSTEM_WINDOWS
#   include <plask/utils/minimal_windows.h>
//#   include <windows.h>
//#   define BOOST_USE_WINDOWS_H
#elif defined(__APPLE__)
#   define PLASK_SYSTEM_MACOS
#   include <mach-o/dyld.h>
#   include <limits.h>
#else
#   include <string>
#   include <limits.h>
#   include <unistd.h>
#endif

namespace plask {

//comes from: http://www.cplusplus.com/forum/general/11104/
std::string exePathAndName() {
#ifdef PLASK_SYSTEM_WINDOWS
    char result[MAX_PATH];
    return std::string(result, GetModuleFileName( NULL, result, MAX_PATH ));
#elif defined(PLASK_SYSTEM_MACOS)
    char result[PATH_MAX];
    uint32_t size = sizeof(result);
    if (_NSGetExecutablePath(result, &size) == 0) {
        return std::string(result);
    } else {
        return std::string();
    }
#else
    char path[PATH_MAX];
    ssize_t count = readlink( "/proc/self/exe", path, PATH_MAX );
    return std::string(path, (count > 0) ? count : 0);
#endif
}

/**
 * Remove from @p dir last path separator and all after.
 * @param dir to changed
 * @return changed @p dir or @p dir if @p dir doesn't include path separator
 */
static std::string dirUp(const std::string& dir) {
    std::string::size_type last_sep = dir.find_last_of(FILE_PATH_SEPARATOR);
    return last_sep == std::string::npos ? dir : dir.substr(0, last_sep);
    //boost::filesystem::absolute(dir).parent_path().string()
}

std::string exePath() {
    return dirUp(exePathAndName());
}

std::string prefixPath() {
    static std::string prefixPath;
    if (!prefixPath.empty()) return prefixPath;
    if (const char* envPath = getenv("PLASK_PREFIX_PATH")) {
        return prefixPath = envPath;
    } else {
        return prefixPath = dirUp(exePath());
    }
}

std::string plaskLibPath() {
    std::string result = prefixPath();
    result += FILE_PATH_SEPARATOR;
    result += "lib";
    result += FILE_PATH_SEPARATOR;
    result += "plask";
    result += FILE_PATH_SEPARATOR;
    return result;
}

std::string plaskSolversPath(const std::string &category) {
    std::string result = plaskLibPath();
    result += "solvers";
    result += FILE_PATH_SEPARATOR;
    result += category;
    result += FILE_PATH_SEPARATOR;
    return result;
}

std::string plaskMaterialsPath() {
    std::string result = plaskLibPath();
    result += "materials";
    result += FILE_PATH_SEPARATOR;
    return result;
}

}   //namespace plask
