#ifndef PLASK__UTILS_OPENMP_H
#define PLASK__UTILS_OPENMP_H

/** @file
This file contains OpenMP related tools.
*/

#include <cstddef>

namespace plask {

/// type similar to std::size_t which can be used in for loop with current version of OpenMP
#ifdef _MSC_VER
typedef std::ptrdiff_t openmp_size_t;  //MSVC support omp only in version 2 which require signed type in for
#define PLASK_PRAGMA(x) __pragma(x)     //MSVC does not implement standard _Pragma but has own extension __pragma
#else
typedef std::size_t openmp_size_t;
#define PLASK_PRAGMA(x) _Pragma(#x)
#endif

}   // namespace plask

#endif // PLASK__UTILS_OPENMP_H
