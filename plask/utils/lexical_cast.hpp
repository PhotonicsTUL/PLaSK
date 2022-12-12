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
    bool negim = false;

    for (i = 0; i < n; ++i) {
        if (src[i] == 'e' || src[i] == 'E') {
            i++; continue;
        }
        if (src[i] == '+')
            break;
        if (src[i] == '-') {
            negim = true; break;
        }
    }

    std::complex<double> result(0., 0.);
    try {
        result.real(boost::lexical_cast<double>(src.substr(0, i)));
        if (i < n) {
            if (i > n-3 || (src[n-1] != 'j' && src[n-1] != 'J'))
                boost::throw_exception(boost::bad_lexical_cast(typeid(std::string), typeid(std::complex<double>)));
            double im = boost::lexical_cast<double>(src.substr(i+1, n-i-2));
            if (negim) im = -im;
            result.imag(im);
        }
    } catch (...) {
         boost::throw_exception(boost::bad_lexical_cast(typeid(std::string), typeid(std::complex<double>)));
    }

    return result;
}


}

#endif // PLASK__UTILS_LEXICAL_CAST_H
