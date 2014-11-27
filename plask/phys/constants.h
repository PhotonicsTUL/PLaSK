#ifndef PLASK__PHYS_CONSTANTS_H
#define PLASK__PHYS_CONSTANTS_H

/**
 * \file
 * This file contains basic physical constants
 */

#include <cmath>

namespace plask {

/**
 * Basic physical quantities and functions
 */
namespace phys {

    constexpr double qe = 1.60217733e-19;       ///< Elementary charge [C]
    constexpr double c = 299792458.;            ///< Speed of light [m/s]
    constexpr double mu0 = 4e-7 * M_PI;         ///< Vacuum permeability [V*s/A/m]
    constexpr double epsilon0 = 1./mu0/c/c;     ///< Vacuum permittivity [F/m]
    constexpr double h_J = 6.62606957e-34;      ///< Planck's constant [J*s]
    constexpr double h_eV = 4.135667516e-15;    ///< Planck's constant [eV*s]
    constexpr double hb_J = 0.5*h_J/M_PI;       ///< Dirac's constant [J*s]
    constexpr double hb_eV = 0.5*h_eV/M_PI;     ///< Dirac's constant [eV*s]
    constexpr double SB = 5.670373e-8;          ///< Stefan-Boltzmann constant [W/m^2/K^4]
    constexpr double kB_J = 1.3806503e-23;      ///< Boltzmann constant [J/K]
    constexpr double kB_eV = 8.6173423e-5;      ///< Boltzmann constant [eV/K]
    constexpr double h_eVc1e9 = 1239.84193009;  ///< h_eV*c*1e9 [eV*m]
    constexpr double me = 9.10938291e-31;       ///< electron mass [kg]

}} // namespace plask::phys

#endif // PLASK__PHYS_CONSTANTS_H
