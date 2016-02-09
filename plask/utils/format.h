#ifndef PLASK__UTILS_FORMAT_H
#define PLASK__UTILS_FORMAT_H

/** @file
This file contains utils to format strings.
*/

//#include <plask/config.h>
#include <boost/lexical_cast.hpp>
#include <cppformat/format.h>

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
 */
inline std::string str(double x) {
    return format("{:.9g}", x);
}

/**
 * Convert complex number to pretty string
 * @param x value to convert
 */
inline std::string str(std::complex<double> x) {
    return format("{:.9g}{:+0.9g}j", real(x), imag(x));
}

}   // namespace plask

#endif // PLASK__FORMAT_UTILS_H
