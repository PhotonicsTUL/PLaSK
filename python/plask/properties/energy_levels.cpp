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

#include "plask/properties/energylevels.hpp"

namespace plask { namespace python {

void register_standard_properties_energy_levels()
{
//     py::class_<EnergyLevels>("EnergyLevels")
//         .def_readonly("electrons", &EnergyLevels::electrons)
//         .def_readonly("heavy_holes", &EnergyLevels::heavy_holes)
//         .def_readonly("light_holes", &EnergyLevels::light_holes)
//     ;
    registerProperty<EnergyLevels,false>();
    py_enum<EnergyLevels::EnumType>()
        .value("ELECTRONS", EnergyLevels::ELECTRONS)
        .value("HEAVY_HOLES", EnergyLevels::HEAVY_HOLES)
        .value("LIGHT_HOLES", EnergyLevels::LIGHT_HOLES)
    ;
}

}} // namespace plask::python
