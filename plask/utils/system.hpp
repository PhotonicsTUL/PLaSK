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
#ifndef PLASK__SYSTEM_H
#define PLASK__SYSTEM_H

/** @file
This file contains portable wrappers for some functions from operating system API.
*/

#include <string>
#include <plask/config.hpp>   //for PLASK_API

namespace plask {

#if defined(_MSC_VER) || defined(__MINGW32__)
constexpr char FILE_PATH_SEPARATOR = '\\';
#else
constexpr char FILE_PATH_SEPARATOR = '/';
#endif

/**
 * Retrieves the fully qualified path for current program executable file.
 * @return program executable name with path
 */
PLASK_API std::string exePathAndName();

/**
 * Get path to current program executable file.
 * @return path to program executable
 */
PLASK_API std::string exePath();

/**
 * Get environment variable PLASK_PREFIX_PATH (should be without rearmost '/' or '\\') or, if it is not set, path one step above the current program executable file.
 * @return path one step above from program executable
 */
PLASK_API std::string prefixPath();

/**
 * Get path to plask library files (shared libraries).
 *
 * This directory contains subdirectories: solvers (see @ref plaskSolversPath), materials (see @ref plaskMaterialsPath).
 * @return path to plask library files (with rearmost '/' or '\\')
 */
PLASK_API std::string plaskLibPath();

/**
 * Get path to files (shared libraries) with solvers in given @p category.
 * @param category name of solvers category
 * @return path (with rearmost '/' or '\\')
 */
PLASK_API std::string plaskSolversPath(const std::string &category);

/**
 * Get path to materials files (shared libraries).
 * @return path (with rearmost '/' or '\\')
 */
PLASK_API std::string plaskMaterialsPath();

}

#endif // SYSTEM_H
