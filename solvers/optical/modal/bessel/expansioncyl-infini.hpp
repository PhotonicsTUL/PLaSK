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
#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H

#include <plask/plask.hpp>

#include "expansioncyl.hpp"
#include "../patterson.hpp"
#include "../meshadapter.hpp"


namespace plask { namespace optical { namespace modal {

struct PLASK_SOLVER_API ExpansionBesselInfini: public ExpansionBessel {

    DataVector<double> kdelts;

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionBesselInfini(BesselSolverCyl* solver);

    /// Perform m-specific initialization
    void init2() override;

    void getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) override;

  protected:

    virtual void integrateParams(Integrals& integrals,
                                 const dcomplex* datap, const dcomplex* datar, const dcomplex* dataz,
                                 dcomplex datap0, dcomplex datar0, dcomplex dataz0) override;

    double fieldFactor(size_t i) override {
        return rbounds[rbounds.size() - 1] / (kpts[i] * kdelts[i]);
    }

    cmatrix getHzMatrix(const cmatrix& Bz, cmatrix& Hz) override { return Bz; }
};

}}} // # namespace plask::optical::modal

#endif // PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
