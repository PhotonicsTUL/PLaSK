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
#include "../python/globals.hpp"
#include "../python/property.hpp"

#include <plask/properties/electrical.hpp>

namespace plask { namespace python {

void register_standard_properties_concentration_carriers(const py::object& flow_module)
{
    registerProperty<CarriersConcentration>(flow_module);
    py_enum<CarriersConcentration::EnumType>()
        .value("MAJORITY", CarriersConcentration::MAJORITY)
        .value("PAIRS", CarriersConcentration::PAIRS)
        .value("ELECTRONS", CarriersConcentration::ELECTRONS)
        .value("HOLES", CarriersConcentration::HOLES)
    ;
}

}} // namespace plask::python
