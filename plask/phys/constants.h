#ifndef PLASK__PHYS_CONSTANTS_H
#define PLASK__PHYS_CONSTANTS_H

/**
 * \file
 * This file contains basic physical constants
 */

namespace plask {

/**
 * Basic physical quantities and functions
 */
namespace phys {

    constexpr double qe = 1.60217733e-19;       ///< elementary charge [C]
    constexpr double c = 299792458.;            ///< speed of light [m/s]
    constexpr double h_J = 6.62606957e-34;      ///< Planck's constant [J*s]
    constexpr double h_eV = 4.135667516e-15;    ///< Planck's constant [eV*s]
    constexpr double SB = 5.670373e-8;          ///< Stefan-Boltzmann constant [W/m^2/K^4]
    constexpr double kB_J = 1.3806503e-23;      ///< Boltzmann constant [J/K]
    constexpr double kB_eV = 8.6173423e-5;      ///< Boltzmann constant [eV/K]

}} // namespace plask::phys

#endif // PLASK__PHYS_CONSTANTS_H
