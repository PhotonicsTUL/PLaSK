#ifndef PLASK__SYSTEM_H
#define PLASK__SYSTEM_H

/** @file
This file includes portable wrappers for some functions from operating system API.
*/

#include <string>

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
std::string exePathAndName();

/**
 * Get path to current program executable file.
 * @return path to program executable
 */
std::string exePath();

/**
 * Get path one step above the current program executable file.
 * @return path one step above from program executable
 */
std::string prefixPath();

/**
 * Get path to plask library files (shared libraries).
 *
 * This directory includes subdirectories: solvers (see @ref getSolversPath), materials (see @ref getMaterialsPath).
 * @return path to plask library files (with rearmost '/' or '\\')
 */
std::string plaskLibPath();

/**
 * Get path to files (shared libraries) with solvers in given @p category.
 * @param category name of solvers category
 * @return path (with rearmost '/' or '\\')
 */
std::string plaskSolversPath(const std::string &category);

std::string plaskMaterialsPath();

}

#endif // SYSTEM_H
