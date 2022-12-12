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
#ifndef PLASK__OPTIONAL_H
#define PLASK__OPTIONAL_H

#include <plask/config.hpp>

#ifdef PLASK_OPTIONAL_STD

#include <optional>
namespace plask {
    using std::optional;
}

#else // PLASK_OPTIONAL_STD
// Use boost::optional

#include <boost/optional.hpp>
namespace plask {
    using boost::optional;
}

#endif // PLASK_SHARED_PTR_STD

#endif // PLASK__OPTIONAL_H
