#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_H

#include <plask/plask.hpp>

#include "../expansion.h"
#include "../meshadapter.h"


namespace plask { namespace solvers { namespace slab {

struct BesselSolverCyl;

struct PLASK_SOLVER_API ExpansionBessel: public Expansion {

    OrderedAxis xmesh;                  ///< Horizontal axis for structure sampling for integration

    size_t N;                           ///< Number of expansion coefficients
    bool initialized;                   ///< Expansion is initialized

//     size_t pil,                         ///< Index of the beginning of the left PML
//            pir;                         ///< Index of the beginning of the right PML

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

    void getMatrices(size_t l, dcomplex k0, dcomplex beta, dcomplex kx, cmatrix& RE, cmatrix& RH) override;

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