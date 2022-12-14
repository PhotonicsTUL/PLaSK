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
#ifndef PLASK__SOLVER_SLAB_EXPANSION_FD2D_H
#define PLASK__SOLVER_SLAB_EXPANSION_FD2D_H

#include <plask/plask.hpp>

#include "../expansion.hpp"
#include "../meshadapter.hpp"

#define INVALID_INDEX size_t(-1)

namespace plask { namespace optical { namespace slab {

struct LinesSolver2D;

struct PLASK_SOLVER_API ExpansionFD2D : public Expansion {
    dcomplex beta,  ///< Longitudinal wavevector [1/µm]
        ktran;      ///< Transverse wavevector [1/µm]

    double left;       ///< Left side of the sampled area
    double right;      ///< Right side of the sampled area
    bool periodic;     ///< Indicates if the geometry is periodic (otherwise use PMLs)
    bool initialized;  ///< Expansion is initialized

    Component symmetry;      ///< Indicates symmetry if `symmetric`
    Component polarization;  ///< Indicates polarization if `separated`

    size_t npl;  ///< Number of the left PML points
    size_t npr;  ///< Number of the right PML points
    size_t ndl;  ///< Number of the left PML distance points
    size_t ndr;  ///< Number of the right PML distance points

    /// Cached permittivity values
    std::vector<DataVector<Tensor3<dcomplex>>> epsilon;

    /// Mesh for getting material data
    shared_ptr<RectangularMesh<2>> material_mesh;

    /// Operational mesh
    shared_ptr<RegularAxis> mesh;

    /// Lateral PML (may be adjusted to match the density)
    PML pml;

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionFD2D(LinesSolver2D* solver);

    /// Indicates if the expansion is a symmetric one
    bool symmetric() const { return symmetry != E_UNSPECIFIED; }

    /// Indicates whether TE and TM modes can be separated
    bool separated() const { return polarization != E_UNSPECIFIED; }

    /**
     * Init expansion
     * \param compute_coeffs compute material coefficients
     */
    void init();

    /// Free allocated memory
    void reset();

    bool diagonalQE(size_t l) const override { return false; }  // 🙁

    size_t matrixSize() const override { return separated() ? mesh->size() : 2 * mesh->size(); }

    void getMatrices(size_t l, cmatrix& RE, cmatrix& RH) override;

    void prepareField() override;

    void cleanupField() override;

    LazyData<Vec<3, dcomplex>> getField(size_t l,
                                        const shared_ptr<const typename LevelsAdapter::Level>& level,
                                        const cvector& E,
                                        const cvector& H) override;

    LazyData<Tensor3<dcomplex>> getMaterialNR(size_t l,
                                              const shared_ptr<const typename LevelsAdapter::Level>& level,
                                              InterpolationMethod interp = INTERPOLATION_DEFAULT) override;

    double integrateField(WhichField field, size_t l, const cvector& E, const cvector& H) override;

    double integratePoyntingVert(const cvector& E, const cvector& H) override;

    void getDiagonalEigenvectors(cmatrix& Te, cmatrix Te1, const cmatrix& RE, const cdiagonal& gamma) override;

  private:
    DataVector<Vec<3, dcomplex>> field;

    inline void checkEdges(size_t i, size_t& im, size_t& ip, Component& sm, Component& sp) {
        size_t N1 = mesh->size() - 1;
        im = i - 1;
        ip = i + 1;
        sm = sp = E_UNSPECIFIED;
        if (i == 0) {
            if (symmetric()) {
                im = 1;
                sm = Component(3 - int(symmetry));
            } else if (periodic) {
                im = N1;
            } else {
                im = INVALID_INDEX;
            }
        }
        if (i == N1) {
            if (periodic) {
                if (symmetric()) {
                    ip = N1 - 1;
                    sp = Component(3 - int(symmetry));
                } else {
                    ip = 0;
                }
            } else {
                ip = INVALID_INDEX;
            }
        }
    }

    template<Component comp>
    inline dcomplex flip(Component sym, dcomplex val) {
        return (sym != comp) ? val : -val;
    }

  protected:
    DataVector<dcomplex> mag;  ///< Magnetic permeability coefficients (used with for PMLs)

    void beforeLayersIntegrals(double lam, double glam) override;

    void layerIntegrals(size_t layer, double lam, double glam) override;

    Tensor3<dcomplex> getEpsilon(const shared_ptr<GeometryD<2>>& geometry,
                                 size_t layer,
                                 double maty,
                                 double lam,
                                 double glam,
                                 size_t j) {
        double T = 0., W = 0., C = 0.;
        for (size_t k = 0, v = j * solver->verts->size(); k != material_mesh->vert()->size(); ++v, ++k) {
            if (solver->stack[k] == layer) {
                double w =
                    (k == 0 || k == material_mesh->vert()->size() - 1) ? 1e-6 : solver->vbounds->at(k) - solver->vbounds->at(k - 1);
                T += w * temperature[v];
                C += w * carriers[v];
                W += w;
            }
        }
        T /= W;
        C /= W;
        Tensor3<dcomplex> nr;
        {
            OmpLockGuard<OmpNestLock> lock;  // this must be declared before `material` to guard its destruction
            auto material = geometry->getMaterial(vec(material_mesh->tran()->at(j), maty));
            lock = material->lock();
            nr = material->NR(lam, T, C);
            if (isnan(nr.c00) || isnan(nr.c11) || isnan(nr.c22) || isnan(nr.c01))
                throw BadInput(solver->getId(), "Complex refractive index (NR) for {} is NaN at lam={}nm, T={}K, n={}/cm3",
                               material->name(), lam, T, C);
        }
        if (nr.c01 != 0.) {
            if (symmetric()) throw BadInput(solver->getId(), "Symmetry not allowed for structure with non-diagonal NR tensor");
            if (separated())
                throw BadInput(solver->getId(), "Single polarization not allowed for structure with non-diagonal NR tensor");
        }
        if (gain_connected && solver->lgained[layer]) {
            auto roles = geometry->getRolesAt(vec(material_mesh->tran()->at(j), maty));
            if (roles.find("QW") != roles.end() || roles.find("QD") != roles.end() || roles.find("gain") != roles.end()) {
                Tensor2<double> g = 0.;
                W = 0.;
                for (size_t k = 0, v = j * solver->verts->size(); k != material_mesh->vert()->size(); ++v, ++k) {
                    if (solver->stack[k] == layer) {
                        double w = (k == 0 || k == material_mesh->vert()->size() - 1)
                                       ? 1e-6
                                       : solver->vbounds->at(k) - solver->vbounds->at(k - 1);
                        g += w * gain[v];
                        W += w;
                    }
                }
                Tensor2<double> ni = glam * g / W * (0.25e-7 / PI);
                nr.c00.imag(ni.c00);
                nr.c11.imag(ni.c00);
                nr.c22.imag(ni.c11);
                nr.c01.imag(0.);
            }
        }
        nr.sqr_inplace();
        return nr;
    }

  public:
    dcomplex getBeta() const { return beta; }
    void setBeta(dcomplex b) {
        if (b != beta) {
            beta = b;
            solver->clearFields();
        }
    }

    dcomplex getKtran() const { return ktran; }
    void setKtran(dcomplex k) {
        if (k != ktran) {
            ktran = k;
            solver->clearFields();
        }
    }

    Component getSymmetry() const { return symmetry; }
    void setSymmetry(Component sym) {
        if (sym != symmetry) {
            symmetry = sym;
            solver->clearFields();
            solver->recompute_integrals = true;
        }
    }

    Component getPolarization() const { return polarization; }
    void setPolarization(Component pol);

    size_t iEx(size_t i) { return 2 * i; }      ///< Get \f$ E_x \f$ index
    size_t iEz(size_t i) { return 2 * i + 1; }  ///< Get \f$ E_z \f$ index
    size_t iHx(size_t i) { return 2 * i + 1; }  ///< Get \f$ H_x \f$ index
    size_t iHz(size_t i) { return 2 * i; }      ///< Get \f$ H_z \f$ index
    size_t iEH(size_t i) { return i; }          ///< Get \f$ E \f$ or \f$ H \f$ index for separated equations

    // dcomplex epszz(size_t l, int i) { return coeffs[l].zz[(i >= 0) ? i : i + nN]; }  ///< Get element of \f$ \varepsilon_{zz} \f$
    // dcomplex epsyy(size_t l, int i) { return coeffs[l].yy[(i >= 0) ? i : i + nN]; }  ///< Get element of \f$ \varepsilon_{yy} \f$
    // dcomplex repsxx(size_t l, int i) {
    //     return coeffs[l].rxx[(i >= 0) ? i : i + nN];
    // }  ///< Get element of \f$ \varepsilon_{xx}^{-1} \f$
    // dcomplex repsyy(size_t l, int i) {
    //     return coeffs[l].ryy[(i >= 0) ? i : i + nN];
    // }  ///< Get element of \f$ \varepsilon_{yy}^{-1} \f$
    // dcomplex epszx(size_t l, int i) { return coeffs[l].zx[(i >= 0) ? i : i + nN]; }  ///< Get element of \f$ \varepsilon_{zx} \f$
    // dcomplex epsxz(size_t l, int i) {
    //     return conj(coeffs[l].zx[(i >= 0) ? i : i + nN]);
    // }  ///< Get element of \f$ \varepsilon_{zx} \f$
    // dcomplex repszx(size_t l, int i) {
    //     return coeffs[l].rzx[(i >= 0) ? i : i + nN];
    // }  ///< Get element of \f$ \varepsilon_{zx}^{-1} \f$
    // dcomplex iepsxz(size_t l, int i) {
    //     return conj(coeffs[l].rzx[(i >= 0) ? i : i + nN]);
    // }                                                              ///< Get element of \f$ \varepsilon_{zx}^{-1} \f$
    // dcomplex muzz(int i) { return mag[(i >= 0) ? i : i + nN]; }    ///< Get element of \f$ \mu_{zz} \f$
    // dcomplex muxx(int i) { return mag[(i >= 0) ? i : i + nN]; }    ///< Get element of \f$ \mu_{xx} \f$
    // dcomplex muyy(int i) { return mag[(i >= 0) ? i : i + nN]; }    ///< Get element of \f$ \mu_{yy} \f$
    // dcomplex rmuzz(int i) { return rmag[(i >= 0) ? i : i + nN]; }  ///< Get element of \f$ \mu_{zz}^{-1} \f$
    // dcomplex rmuxx(int i) { return rmag[(i >= 0) ? i : i + nN]; }  ///< Get element of \f$ \mu_{xx}^{-1} \f$
    // dcomplex rmuyy(int i) { return rmag[(i >= 0) ? i : i + nN]; }  ///< Get element of \f$ \mu_{yy}^{-1} \f$

    dcomplex repsyy(size_t l, int i) const { return epsilon[l][i].c22; }
    dcomplex epsxx(size_t l, int i) const { return epsilon[l][i].c00; }
    dcomplex epszz(size_t l, int i) const { return epsilon[l][i].c11; }
    dcomplex epszx(size_t l, int i) const { return epsilon[l][i].c01; }
    dcomplex epsxz(size_t l, int i) const { return conj(epsilon[l][i].c01); }

    dcomplex rmuyy(size_t i) const { return 1. / mag[i]; }
    dcomplex muxx(size_t i) const { return 1. / mag[i]; }
    dcomplex muzz(size_t i) const { return mag[i]; }
};

}}}  // namespace plask::optical::slab

#endif  // PLASK__SOLVER_SLAB_EXPANSION_FD2D_H
