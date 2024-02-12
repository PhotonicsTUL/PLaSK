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
#include "../python_globals.hpp"
#include "../python_property.hpp"

#include "plask/properties/gain.hpp"

namespace plask { namespace python {

/**
 * Register standard gain properties to Python.
 *
 * Add new gain properties here
 */
void register_standard_properties_gain()
{
    registerProperty<Gain>();
    py_enum<Gain::EnumType>()
        .value("", Gain::GAIN)
        .value("CONC", Gain::DGDN)
    ;

    registerProperty<Luminescence>();
}

}} // namespace plask::python
