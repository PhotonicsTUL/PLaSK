#ifndef PLASK__UTILS_LEXICAL_CAST_H
#define PLASK__UTILS_LEXICAL_CAST_H

/** \file
This file contains lexical cast for std::complex<double>
*/

#include <complex>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/trim.hpp>

namespace boost {

template <>
inline std::complex<double> lexical_cast(const std::string &arg)
{
    std::string src = boost::algorithm::trim_right_copy(arg);

    size_t n = src.length(), i;

    for (i = 0; i < n; ++i) {
        if (src[i] == 'e' || src[i] == 'E') {
            i++; continue;
        }
        if (src[i] == '+' || src[i] == '-') break;
    }

    std::complex<double> result(0., 0.);
    try {
        result.real(boost::lexical_cast<double>(src.substr(0, i)));
        if (i < n) {
            if (src[n-1] != 'j' && src[n-1] != 'J')
                boost::throw_exception(boost::bad_lexical_cast(typeid(std::string), typeid(std::complex<double>)));
            result.imag(boost::lexical_cast<double>(src.substr(i, n-i-1)));
        }
    } catch (...) {
         boost::throw_exception(boost::bad_lexical_cast(typeid(std::string), typeid(std::complex<double>)));
    }

    return result;
}


}

#endif // PLASK__UTILS_LEXICAL_CAST_H
