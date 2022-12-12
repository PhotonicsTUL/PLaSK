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
#include "thermal.hpp"
#include "electrical.hpp"
#include "gain.hpp"
#include "optical.hpp"
#include "energylevels.hpp"

namespace plask {

constexpr const char* Temperature::NAME;                        constexpr const char* Temperature::UNIT;
constexpr const char* HeatFlux::NAME;                           constexpr const char* HeatFlux::UNIT;
constexpr const char* Heat::NAME;                               constexpr const char* Heat::UNIT;
constexpr const char* ThermalConductivity::NAME;                constexpr const char* ThermalConductivity::UNIT;

constexpr const char* Voltage::NAME;                            constexpr const char* Voltage::UNIT;
constexpr const char* Potential::NAME;                          constexpr const char* Potential::UNIT;
constexpr const char* CurrentDensity::NAME;                     constexpr const char* CurrentDensity::UNIT;
constexpr const char* CarriersConcentration::NAME;              constexpr const char* CarriersConcentration::UNIT;
constexpr const char* Conductivity::NAME;                       constexpr const char* Conductivity::UNIT;
constexpr const char* FermiLevels::NAME;                        constexpr const char* FermiLevels::UNIT;
constexpr const char* BandEdges::NAME;                          constexpr const char* BandEdges::UNIT;
constexpr const char* EnergyLevels::NAME;                       constexpr const char* EnergyLevels::UNIT;

constexpr const char* Gain::NAME;                               constexpr const char* Gain::UNIT;
constexpr const char* Luminescence::NAME;                       constexpr const char* Luminescence::UNIT;

constexpr const char* RefractiveIndex::NAME;                    constexpr const char* RefractiveIndex::UNIT;
constexpr const char* LightMagnitude::NAME;                     constexpr const char* LightMagnitude::UNIT;
constexpr const char* LightE::NAME;                             constexpr const char* LightE::UNIT;
constexpr const char* LightH::NAME;                             constexpr const char* LightH::UNIT;
constexpr const char* ModeLightMagnitude::NAME;                 constexpr const char* ModeLightMagnitude::UNIT;
constexpr const char* ModeLightE::NAME;                         constexpr const char* ModeLightE::UNIT;
constexpr const char* ModeLightH::NAME;                         constexpr const char* ModeLightH::UNIT;
constexpr const char* ModeWavelength::NAME;                     constexpr const char* ModeWavelength::UNIT;
constexpr const char* ModeLoss::NAME;                           constexpr const char* ModeLoss::UNIT;
constexpr const char* ModePropagationConstant::NAME;            constexpr const char* ModePropagationConstant::UNIT;
constexpr const char* ModeEffectiveIndex::NAME;                 constexpr const char* ModeEffectiveIndex::UNIT;

}
