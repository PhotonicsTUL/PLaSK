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

    const double qe = 1.60217733e-19;       ///< elementary charge [C]
    const double c = 299792458.;            ///< speed of light [m/s]
    const double h_J = 6.62606957e-34;      ///< Planck's constant [J*s]
    const double h_eV = 4.135667516e-15;    ///< Planck's constant [J*s]
    const double SB = 5.670373e-8;          ///< Stefan-Boltzmann constant [W/m^2/K^4]
    const double kB_J = 1.3806503e-23;      ///< Boltzmann constant [J/K]
    const double kB_eV = 8.6173423e-5;      ///< Boltzmann constant [eV/K]

}} // namespace plask::phys

#endif // PLASK__PHYS_CONSTANTS_H
