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
#ifndef PLASK__SOLVER_SLAB_IMPEDANCE_H
#define PLASK__SOLVER_SLAB_IMPEDANCE_H

#include "matrices.hpp"
#include "xance.hpp"
#include "solver.hpp"


namespace plask { namespace optical { namespace modal {

/**
 * Base class for all solvers using reflection matrix method.
 */
struct PLASK_SOLVER_API ImpedanceTransfer: public XanceTransfer {

    cvector getReflectionVector(const cvector& incident, IncidentDirection side) override;

    ImpedanceTransfer(ModalBase* solver, Expansion& expansion);

  protected:

    void getFinalMatrix() override;

    void determineFields() override;

    // cvector getReflectionVectorH(const cvector& incident, IncidentDirection side);

    void determineReflectedFields(const cvector& incident, IncidentDirection side) override;

    /**
     * Find impedance matrix for the part of the structure
     * \param start starting layer
     * \param end last layer (reflection matrix is computed for this layer)
     */
    void findImpedance(std::ptrdiff_t start, std::ptrdiff_t end);
};


}}} // namespace plask::optical::modal

#endif // PLASK__SOLVER_SLAB_IMPEDANCE_H
