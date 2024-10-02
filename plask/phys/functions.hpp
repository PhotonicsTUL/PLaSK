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
#ifndef PLASK__PHYS_FUNCTIONS_H
#define PLASK__PHYS_FUNCTIONS_H

#include <plask/config.hpp>  //for PLASK_API
#include "constants.hpp"

namespace plask { namespace phys {

/**
 * Varshni functions
 * \param Eg0K [eV] bandgap at 0K
 * \param alpha [eV/K] Varshni parameter α
 * \param beta [K] Varshni parameter β
 * \param T [K] temperature
 * \return [eV] bandgap at temperature T
 */
PLASK_API double Varshni(double Eg0K, double alpha, double beta, double T);

/**
 * Convert wavelength to photon energy
 * \param lam [nm] wavelength
 * \return [eV] photon energy
 */
inline static double nm_to_eV(double wavelength) { return h_eVc1e9 / wavelength; }

/**
 * Convert photon energy to wavelength
 * \param E [eV] photon energy
 * \return [nm] wavelength
 */
inline static double eV_to_nm(double E) { return h_eVc1e9 / E; }

}}  // namespace plask::phys

#endif  // PLASK__PHYS_FUNCTIONS_H
