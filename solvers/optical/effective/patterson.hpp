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
#ifndef PLASK__PATTERSON_H
#define PLASK__PATTERSON_H

#include <plask/plask.hpp>

#include "patterson-data.hpp"

/**
 * Patterson quadrature for complex function along specified line
 */

namespace plask { namespace optical { namespace effective {

/**
 * Compute Patterson quadrature along line a-b in complex plane with specified precision
 * \param fun function to integrate
 * \param a starting point
 * \param b final point
 * \param[in,out] err on input maximum error, on output estimated error
 * \return computed integral
 **/
template <typename S, typename T>
S patterson(const std::function<S(T)>& fun, T a, T b, double& err);


}}} // namespace plask::optical::effective

#endif // PLASK__PATTERSON_H