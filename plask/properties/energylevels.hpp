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
#ifndef PLASK__ENERGYLEVELS_H
#define PLASK__ENERGYLEVELS_H

#include <vector>

#include <plask/provider/providerfor.hpp>

namespace plask {

/**
 * Energy levels for electrons and holes (eV)
 */
struct PLASK_API EnergyLevels: public MultiFieldProperty<std::vector<double>> {
    enum EnumType {
        ELECTRONS,
        HEAVY_HOLES,
        LIGHT_HOLES
    };
    static constexpr size_t NUM_VALS = 3;
    static constexpr const char* NAME = "energy levels for electrons and holes";
    static constexpr const char* UNIT = "eV";
};

} // namespace plask

#endif // PLASK__ENERGYLEVELS_H
