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
#ifndef PLASK__UTILS_FORMAT_H
#define PLASK__UTILS_FORMAT_H

/** @file
This file contains utils to format strings.
*/

#include <complex>

//#include <plask/config.hpp>
#include <boost/lexical_cast.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace plask {

using fmt::format;

/**
 * Convert something to pretty string
 * @param x value to convert
 */
template <typename T>
inline std::string str(T x) {
    return boost::lexical_cast<std::string>(x);
}

/**
 * Convert double number to pretty string
 * @param x value to convert
 * @param fmt format to use
 */
inline std::string str(double x, const char* fmt="{:.9g}") {
    return format(fmt, x);
}

/**
 * Convert complex number to pretty string
 * @param x value to convert
 * @param fmt format to use
 * @param rfmt format used if Im(x) == 0
 */
inline std::string str(std::complex<double> x, const char* fmt="{:.9g}{:+0.9g}j", const char* rfmt=nullptr) {
    if (!rfmt || imag(x) != 0.)
        return format(fmt, real(x), imag(x));
    else
        return format(rfmt, real(x));
}

}   // namespace plask

#endif // PLASK__FORMAT_UTILS_H
