#ifndef PLASK__UTILS_FORMAT_H
#define PLASK__UTILS_FORMAT_H

/** @file
This file includes utils to format strings.
*/

#include <boost/format.hpp> //TODO maybe better to take simple code from http://en.wikipedia.org/wiki/Variadic_templates

namespace plask {

///Recursion end of format_add_args. Do nothing.
inline void format_add_args(boost::format&) {}

/**
 * Call: @code format % first_arg; format_add_args(format, rest_args...); @endcode
 * @param format, first_arg, rest_args argument for boost format calls
 * @see http://www.boost.org/doc/libs/1_48_0/libs/format/
 */
template <typename firstT, typename... restT>
inline void format_add_args(boost::format& format, const firstT& first_arg, const restT&... rest_args) {
    format % first_arg;
    format_add_args(format, rest_args...);
}

/**
 * Format string using boost format.
 * @param format template string which have %1%, %2%, ...
 * @param args arguments for %1%, %2%, ...
 * @return formated string
 * @see http://www.boost.org/doc/libs/1_48_0/libs/format/
 */
template <typename... T>
std::string format(const std::string& msg, const T&... args) {
    boost::format format;
    format_add_args(format, args...);
    return format.str();
}

}   // namespace plask

#endif // PLASK__FORMAT_UTILS_H
