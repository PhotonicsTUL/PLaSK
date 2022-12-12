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
#ifndef PLASK__UTILS_OPENMP_H
#define PLASK__UTILS_OPENMP_H

/** @file
This file contains OpenMP related tools.
*/

#include <cstddef>

#include "warnings.hpp"

namespace plask {

/// type similar to std::size_t which can be used in for loop with current version of OpenMP
#ifdef _MSC_VER
typedef std::ptrdiff_t openmp_size_t;  //MSVC support omp only in version 2 which require signed type in for
#else
typedef std::size_t openmp_size_t;
#endif

}   // namespace plask

#endif // PLASK__UTILS_OPENMP_H
