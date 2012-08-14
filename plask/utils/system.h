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

}

#endif // SYSTEM_H
