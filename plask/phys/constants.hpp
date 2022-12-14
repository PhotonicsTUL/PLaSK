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
#ifndef PLASK__PHYS_CONSTANTS_H
#define PLASK__PHYS_CONSTANTS_H

/**
 * \file
 * This file contains basic physical constants
 */

#include "../math.hpp"

namespace plask {

/**
 * Basic physical quantities and functions
 */
namespace phys {

    constexpr double qe = 1.60217733e-19;       ///< Elementary charge [C]
    constexpr double me = 9.10938291e-31;       ///< Electron mass [kg]
    constexpr double c = 299792458.;            ///< Speed of light [m/s]
    constexpr double mu0 = 4e-7 * PI;           ///< Vacuum permeability [V*s/A/m]
    constexpr double epsilon0 = 1./mu0/c/c;     ///< Vacuum permittivity [F/m]
    constexpr double Z0 = 376.73031346177066;   ///< Free space admittance [Ω]
    constexpr double h_J = 6.62606957e-34;      ///< Planck's constant [J*s]
    constexpr double h_eV = 4.135667516e-15;    ///< Planck's constant [eV*s]
    constexpr double hb_J = 0.5*h_J/PI;         ///< Dirac's constant [J*s]
    constexpr double hb_eV = 0.5*h_eV/PI;       ///< Dirac's constant [eV*s]
    constexpr double SB = 5.670373e-8;          ///< Stefan-Boltzmann constant [W/m^2/K^4]
    constexpr double kB_J = 1.3806503e-23;      ///< Boltzmann constant [J/K]
    constexpr double kB_eV = 8.6173423e-5;      ///< Boltzmann constant [eV/K]
    constexpr double h_eVc1e9 = 1239.84193009;  ///< h_eV*c*1e9 [eV*m]

}} // namespace plask::phys

#endif // PLASK__PHYS_CONSTANTS_H
