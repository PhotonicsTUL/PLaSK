#ifndef PLASK__UTILS_FORMAT_H
#define PLASK__UTILS_FORMAT_H

#include <boost/lexical_cast.hpp>

/** @file
This file contains utils to format strings.
*/

#include <boost/format.hpp> //TODO maybe better to take simple code from http://en.wikipedia.org/wiki/Variadic_templates

#include "../math.h"

namespace plask {

///Recursion end of format_add_args. Do nothing.
inline void format_add_args(boost::format&) {}

/**
 * Call: @code format % first_arg; format_add_args(format, rest_args...); @endcode
 * @param format, first_arg, rest_args argument for boost format calls
 * @see http://www.boost.org/doc/libs/1_48_0/libs/format/
 */
template <typename firstT, typename... restT>
inline void format_add_args(boost::format& format, firstT&& first_arg, restT&&... rest_args) {
    format % std::forward<firstT>(first_arg);
    format_add_args(format, std::forward<restT>(rest_args)...);
}

/**
 * Format string using boost format.
 * @param msg template string which have %1%, %2%, ...
 * @param args arguments for %1%, %2%, ...
 * @return formated string
 * @see http://www.boost.org/doc/libs/1_48_0/libs/format/
 */
template <typename... T>
std::string format(const std::string& msg, T&&... args) {
    boost::format format(msg);
    format_add_args(format, std::forward<T>(args)...);
    return format.str();
}


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
    return format("%.9g", x);
}

/**
 * Convert complex number to pretty string
 * @param x value to convert
 */
inline std::string str(dcomplex x) {
    return format("%.9g%+-0.9gj", real(x), imag(x));
}



}   // namespace plask

#endif // PLASK__FORMAT_UTILS_H
