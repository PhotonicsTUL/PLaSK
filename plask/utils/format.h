#ifndef PLASK__FORMAT_H
#define PLASK__FORMAT_H

#include <boost/format.hpp> //TODO maybe better to take simple code from http://en.wikipedia.org/wiki/Variadic_templates

namespace plask {

//recursion end
inline void format_add_args(boost::format& format) {}

template <typename firstT, typename... restT>
inline void format_add_args(boost::format& format, const firstT& first_arg, const restT&... rest_args) {
    format % first_arg;
    format_add_args(format, rest_args...);
}

template <typename... T>
std::string format(const std::string& msg, const T&... args) {
    boost::format format;
    format_add_args(format, args...);
    return format.str();
}

}   // namespace plask

#endif // FORMAT_H
