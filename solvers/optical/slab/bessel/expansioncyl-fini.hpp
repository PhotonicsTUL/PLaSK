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
#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_FINI_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_FINI_H

#include <plask/plask.hpp>

#include "../meshadapter.hpp"
#include "../patterson.hpp"
#include "expansioncyl.hpp"

namespace plask { namespace optical { namespace slab {

struct PLASK_SOLVER_API ExpansionBesselFini : public ExpansionBessel {
    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionBesselFini(BesselSolverCyl* solver);

    /// Fill kpts with Bessel zeros
    void computeBesselZeros();

    /// Perform m-specific initialization
    void init2() override;

    /// Free allocated memory
    void reset() override;

    void getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) override;

  protected:
    /// Integrals for magnetic permeability
    Integrals mu_integrals;

    double fieldFactor(size_t i) override;

    cmatrix getHzMatrix(const cmatrix& Bz, cmatrix& Hz) override {
        mult_matrix_by_matrix(mu_integrals.V_k, Bz, Hz);
        return Hz;
    }

    virtual void integrateParams(Integrals& integrals,
                                 const dcomplex* datap, const dcomplex* datar, const dcomplex* dataz,
                                 dcomplex datap0, dcomplex datar0, dcomplex dataz0) override;

    #ifndef NDEBUG
      public:
        cmatrix muV_k();
        cmatrix muTss();
        cmatrix muTsp();
        cmatrix muTps();
        cmatrix muTpp();
    #endif
};

}}}  // namespace plask::optical::slab

#endif  // PLASK__SOLVER__SLAB_EXPANSIONCYL_FINI_H
