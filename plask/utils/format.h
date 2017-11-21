#ifndef PLASK__UTILS_FORMAT_H
#define PLASK__UTILS_FORMAT_H

/** @file
This file contains utils to format strings.
*/

//#include <plask/config.h>
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
