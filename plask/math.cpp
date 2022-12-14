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
#include "math.hpp"

namespace plask {

template std::complex<double> parse_complex<double>(std::string str_to_parse);

AccurateSum &AccurateSum::operator=(double v) {
    s = v;
    c = 0.0;
    return *this;
}

//Kahan and Babuska summation, Neumaier variant.
//(better than Kahan method? https://github.com/JuliaLang/julia/issues/199)
/*AccurateSum& AccurateSum::operator+=(double v) {
    const double t = s + v;
    if (std::abs(s) >= std::abs(v))
        c += (s - t) + v;
    else
        c += (v - t) + s;
    s = t;
    return *this;
}

operator double() const { return s + c; }

AccurateSum& AccurateSum::operator+=(const AccurateSum& other) {
    *this += other.s;
    *this += other.c;
    return *this;
}*/

AccurateSum &AccurateSum::operator+=(const AccurateSum &other) {
    *this += -other.c;
    *this += other.s;
    return *this;
}

AccurateSum &AccurateSum::operator+=(double v) {
    const double y = v - c;    // compensed value to add
    const double news = s + y; // new sum
    c = (news - s) - y;        // news is to hight by c (algebraically, c should always be zero, typically c is negative)
    s = news;                  // we will substract c from sum next time
    return *this;
}

}   // namespace plask
