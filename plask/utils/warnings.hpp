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
#ifndef PLASK_WARNINGS_H
#define PLASK_WARNINGS_H

/** @file
This file contains portable utils to manage (mainly disable for a given fragment of code) compiler warnings.
*/

#ifdef _MSC_VER
    #define PLASK_PRAGMA(x) __pragma(x)     // MSVC does not implement standard _Pragma but has own extension __pragma
#else
    /// allows for putting pragma from macro
    #define PLASK_PRAGMA(x) _Pragma(#x)
#endif

// this helps disable warning about unusing parameter in doxygen-friendly way
#ifdef DOXYGEN
    /// mark that the function parameter is unused and prevent compiler from warning about it
    #define PLASK_UNUSED(arg) arg
#else
    #define PLASK_UNUSED(arg)
#endif



#ifdef _MSC_VER // ----------- Visual C++ -----------

#define PLASK_NO_CONVERSION_WARNING_BEGIN \
    PLASK_PRAGMA(warning(push))   \
    PLASK_PRAGMA(warning(disable: 4244))

#define PLASK_NO_UNUSED_VARIABLE_WARNING_BEGIN \
    PLASK_PRAGMA(warning(push))
    // TODO może C4101 ?

#define PLASK_NO_WARNING_END \
    PLASK_PRAGMA(warning(pop))



#elif defined(__GNUC__) // ----------- GNU C++ -----------

#define PLASK_NO_CONVERSION_WARNING_BEGIN \
    PLASK_PRAGMA(GCC diagnostic push) \
    PLASK_PRAGMA(GCC diagnostic ignored "-Wconversion")

#define PLASK_NO_UNUSED_VARIABLE_WARNING_BEGIN \
    PLASK_PRAGMA(GCC diagnostic push) \
    PLASK_PRAGMA(GCC diagnostic ignored "-Wunused-variable")

#define PLASK_NO_WARNING_END \
    PLASK_PRAGMA(GCC diagnostic pop)



#else   // ----------- unknown compiler -----------

/// beginning from the place this macro is put, disable warnings about possibly danger conversions (with precision loss, etc.)
#define PLASK_NO_CONVERSION_WARNING_BEGIN

/// beginning from the place this macro is put, disable warnings about unused variables
#define PLASK_NO_UNUSED_VARIABLE_WARNING_BEGIN

/// ends the fragment where some warnings were disabled by PLASK_NO_*_WARNING_BEGIN macro (must be used twice to end two successive PLASK_NO_*_WARNING_BEGIN)
#define PLASK_NO_WARNING_END

#endif

#endif // PLASK_WARNINGS_H
