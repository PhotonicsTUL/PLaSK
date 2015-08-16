#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_H

#include <plask/plask.hpp>

#include "../expansion.h"
#include "../patterson.h"
#include "../meshadapter.h"

namespace plask { namespace solvers { namespace slab {

struct BesselSolverCyl;

struct PLASK_SOLVER_API ExpansionBessel: public Expansion {

    size_t N;                           ///< Number of expansion coefficients
    bool initialized;                   ///< Expansion is initialized

//     size_t pil,                         ///< Index of the beginning of the left PML
//            pir;                         ///< Index of the beginning of the right PML

    /// Horizontal axis with separate integration intervals.
    /// material functions contain discontinuities at these points
    OrderedAxis xbounds;                
    
    /// Matrices with computed integrals necessary to construct RE and RH matrices
    class Integrals {
        cvector iem;    ///< J_{m-1}(gr) eps^{-1}(r) J_{m-1}(kr) r dr
        cvector iep;    ///< J_{m+1}(gr) eps^{-1}(r) J_{m+1}(kr) r dr
        cvector dem;    ///< J_{m-1}(gr) deps/dr J_{m}(kr) r dr
        cvector dep;    ///< J_{m+1}(gr) deps/dr J_{m}(kr) r dr
        cvector em;     ///< J_{m-1}(gr) eps(r) J_{m-1}(kr) r dr
        cvector ep;     ///< J_{m+1}(gr) eps(r) J_{m+1}(kr) r dr
        inline size_t idx(size_t i, size_t j) const { return (i<=j)? j*(j+1)/2 + i: i*(i+1)/2 + j; }
      public:
        Integrals() {}
        Integrals(size_t N) { reset(N); }
        void reset(size_t N) {
            size_t len = N*(N+1)/2;
            iem.reset(len);
            iep.reset(len);
            dem.reset(len);
            dep.reset(len);
            em.reset(len);
            ep.reset(len);
        }
        dcomplex& ieps_minus(size_t i, size_t j) { return iem[idx(i,j)]; }
        const dcomplex& ieps_minus(size_t i, size_t j) const { return iem[idx(i,j)]; }
        dcomplex& ieps_plus(size_t i, size_t j) { return iep[idx(i,j)]; }
        const dcomplex& ieps_plus(size_t i, size_t j) const { return iep[idx(i,j)]; }
        dcomplex& deps_minus(size_t i, size_t j) { return dem[idx(i,j)]; }
        const dcomplex& deps_minus(size_t i, size_t j) const { return dem[idx(i,j)]; }
        dcomplex& deps_plus(size_t i, size_t j) { return dep[idx(i,j)]; }
        const dcomplex& deps_plus(size_t i, size_t j) const { return dep[idx(i,j)]; }
        dcomplex& eps_minus(size_t i, size_t j) { return em[idx(i,j)]; }
        const dcomplex& eps_minus(size_t i, size_t j) const { return em[idx(i,j)]; }
        dcomplex& eps_plus(size_t i, size_t j) { return ep[idx(i,j)]; }
        const dcomplex& eps_plus(size_t i, size_t j) const { return ep[idx(i,j)]; }
    };

    /// Computed integrals
    std::vector<Integrals> integrals;
    
    /// Information if the layer is diagonal
    std::vector<bool> diagonals;
    
    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionBessel(BesselSolverCyl* solver);

    /**
     * Init expansion
     * \param compute_coeffs compute material coefficients
     */
    void init();

    /// Free allocated memory
    void reset();

    /// Compute itegrals for RE and RH matrices
    void computeIntegrals() {
        size_t nlayers = lcount();
//TODO         assert(coeffs.size() == nlayers);
        for (size_t l = 0; l < nlayers; ++l)
            layerIntegrals(l);
    }

    size_t lcount() const override;

    bool diagonalQE(size_t l) const override {
        return diagonals[l];
    }

    size_t matrixSize() const override { return 2*N; } // TODO should be N for m = 0?

    void getMatrices(size_t l, cmatrix& RE, cmatrix& RH) override;

    void prepareField() override;

    void cleanupField() override;

    DataVector<const Vec<3,dcomplex>> getField(size_t l,
                                               const shared_ptr<const typename LevelsAdapter::Level>& level,
                                               const cvector& E, const cvector& H) override;

    LazyData<Tensor3<dcomplex>> getMaterialNR(size_t l,
                                              const shared_ptr<const typename LevelsAdapter::Level>& level,
                                              InterpolationMethod interp=INTERPOLATION_DEFAULT) override;

  protected:

    /**
     * Compute itegrals for RE and RH matrices
     * \param l layer number
     */
    void layerIntegrals(size_t l);

  public:

};

}}} // # namespace plask::solvers::slab

#endif // PLASK__SOLVER__SLAB_EXPANSIONCYL_H