/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2026 Lodz University of Technology
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
#ifndef PLASK_PROPERTIES_CAPACITANCE_HPP
#define PLASK_PROPERTIES_CAPACITANCE_HPP

#include <plask/provider/providerfor.hpp>

namespace plask {

/**
 * AC voltage (V)
 */
struct PLASK_API AcVoltage: public FieldProperty<dcomplex> {
    static constexpr const char* NAME = "AC voltage amplitude";
    static constexpr const char* UNIT = "V";
};

/**
 * AC current density (kA/cm²)
 * This is 2D vector for two-dimensional sovers
 */
struct PLASK_API AcCurrentDensity: public VectorFieldProperty<dcomplex> {
    static constexpr const char* NAME = "AC current density amplitude";
    static constexpr const char* UNIT = "kA/cm²";
};

} // namespace plask

#endif // PLASK_PROPERTIES_CAPACITANCE_HPP
