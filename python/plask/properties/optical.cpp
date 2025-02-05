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

#include <plask/properties/optical.hpp>

namespace plask { namespace python {

/**
 * Register standard optical properties to Python.
 *
 * Add new optical properties here
 */
void register_standard_properties_optical(const py::object& flow_module)
{
    registerProperty<LightMagnitude>(flow_module);
    registerProperty<LightE>(flow_module);
    registerProperty<LightH>(flow_module);

    registerProperty<ModeLightMagnitude>(flow_module);
    //TODO RegisterCombinedProvider<LightMagnitudeSumProvider<Geometry2DCartesian>>("SumOfLightMagnitude", flow_module);
    registerProperty<ModeLightE>(flow_module);
    registerProperty<ModeLightH>(flow_module);

    registerProperty<ModeWavelength>(flow_module);
    registerProperty<ModeLoss>(flow_module);
    registerProperty<ModePropagationConstant>(flow_module);
    registerProperty<ModeEffectiveIndex>(flow_module);
}

}} // namespace plask::python
