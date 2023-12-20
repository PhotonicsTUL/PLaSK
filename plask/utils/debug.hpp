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
#ifndef PLASK__UTILS_DEBUG_H
#define PLASK__UTILS_DEBUG_H

/** @file
This file contains debugging utils. For internal use only.
*/

#include <plask/config.hpp>

#if defined(PRINT_STACKTRACE_ON_EXCEPTION) && !(defined(_WIN32) || defined(__WIN32__) || defined(WIN32))

#include <backward.hpp>

#define PLASK_PRINT_STACK_HERE(d) { backward::StackTrace st; st.load_here(d); backward::Printer p; p.print(st); }

#else

#define PLASK_PRINT_STACK_HERE(d)

#endif

#endif // PLASK__UTILS_DEBUG_H
