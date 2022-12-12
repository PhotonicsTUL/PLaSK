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

#include <plask/config.hpp>   //for PLASK_API

namespace plask { namespace phys {

    /**
     * Varshni functions
     * TODO doc
     * \param Eg0K [eV]
     * \param alpha [eV/K]
     * \param beta [K]
     * \param T [K]
     * \return [eV]
     */
    PLASK_API double Varshni(double Eg0K, double alpha, double beta, double T);

    /**
     * Energy of Photon
     * TODO doc
     * \param lam [nm]
     * \return [eV]
     */
    PLASK_API double PhotonEnergy(double lam);

}} // namespace plask::phys

#endif // PLASK__PHYS_FUNCTIONS_H
