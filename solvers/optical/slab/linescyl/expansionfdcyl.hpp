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
#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_H

#include <plask/plask.hpp>

#include "../expansion.hpp"
#include "../meshadapter.hpp"

namespace plask { namespace optical { namespace slab {

struct LinesSolverCyl;

struct PLASK_SOLVER_API ExpansionLines : public Expansion {
    int m;  ///< Angular dependency index

    bool initialized;  ///< Expansion is initialized

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionLines(LinesSolverCyl* solver);

    virtual ~ExpansionLines() {}

    /// Init expansion
    void init();

    /// Free allocated memory
    virtual void reset();

    bool diagonalQE(size_t l) const override { return diagonals[l]; }

    size_t matrixSize() const override;

    void prepareField() override;

    void cleanupField() override;

    LazyData<Vec<3, dcomplex>> getField(size_t layer,
                                        const shared_ptr<const typename LevelsAdapter::Level>& level,
                                        const cvector& E,
                                        const cvector& H) override;

    LazyData<Tensor3<dcomplex>> getMaterialNR(size_t layer,
                                              const shared_ptr<const typename LevelsAdapter::Level>& level,
                                              InterpolationMethod interp = INTERPOLATION_DEFAULT) override;

  protected:
    /// The real mesh
    shared_ptr<MeshAxis> raxis;

    /// Information if the layer is diagonal
    std::vector<bool> diagonals;

    /// Obtained temperature
    LazyData<double> temperature;

    /// Flag indicating if the gain is connected
    bool gain_connected;

    /// Obtained gain
    LazyData<Tensor2<double>> gain;

    std::vector<DataVector<Tensor3<dcomplex>>> epsilons;
    DataVector<Tensor3<dcomplex>> mu;

    void beforeLayersIntegrals(double lam, double glam) override;

    void afterLayersIntegrals() override;

    void layerIntegrals(size_t layer, double lam, double glam) override;

    void getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) override;

  public:
    double integratePoyntingVert(const cvector& E, const cvector& H) override;

    unsigned getM() const { return m; }
    void setM(unsigned n) {
        if (int(n) != m) {
            write_debug("{0}: m changed from {1} to {2}", solver->getId(), m, n);
            m = n;
            solver->recompute_integrals = true;
            solver->clearFields();
        }
    }

    /// Get \f$ E_r \f$ index
    size_t iEr(size_t i) { return 2 * i - ((m == 1) ? 0 : 2); }

    /// Get \f$ E_φ \f$ index
    size_t iEp(size_t i) { return 2 * i - 1; }

    /// Get \f$ E_r \f$ index
    size_t iHp(size_t i) { return 2 * i - 1; }

    /// Get \f$ E_φ \f$ index
    size_t iHr(size_t i) { return 2 * i - ((m == 1) ? 0 : 2); }

    /// Shift between adjacent indices
    static constexpr int i1 = 2;
};

}}}  // namespace plask::optical::slab

#endif  // PLASK__SOLVER__SLAB_EXPANSIONCYL_H
