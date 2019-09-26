#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_H

#include <plask/plask.hpp>

#include "../expansion.h"
#include "../meshadapter.h"

namespace plask { namespace optical { namespace slab {

struct CylindersSolverCyl;

struct PLASK_SOLVER_API ExpansionCylinders: public Expansion {

    int m;                              ///< Angular dependency index

    bool initialized;                   ///< Expansion is initialized

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionCylinders(CylindersSolverCyl* solver);

    virtual ~ExpansionCylinders() {}

    /// Init expansion
    void init();

    /// Free allocated memory
    virtual void reset();

    bool diagonalQE(size_t l) const override {
        return diagonals[l];
    }

    size_t matrixSize() const override;

    void prepareField() override;

    void cleanupField() override;

    LazyData<Vec<3,dcomplex>> getField(size_t layer,
                                       const shared_ptr<const typename LevelsAdapter::Level>& level,
                                       const cvector& E, const cvector& H) override;

    LazyData<Tensor3<dcomplex>> getMaterialNR(size_t layer,
                                              const shared_ptr<const typename LevelsAdapter::Level>& level,
                                              InterpolationMethod interp=INTERPOLATION_DEFAULT) override;

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

    void prepareIntegrals(double lam, double glam) override;

    void cleanupIntegrals(double, double) override;

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
    size_t iEs(size_t i) { return 2 * i; }

    /// Get \f$ E_φ \f$ index
    size_t iEp(size_t i) { return 2 * i + 1; }

    /// Get \f$ E_r \f$ index
    size_t iHs(size_t i) { return 2 * i + 1; }

    /// Get \f$ E_φ \f$ index
    size_t iHp(size_t i) { return 2 * i; }

    /// Shift between adjacent indices
    static constexpr int i1 = 2;
};

}}} // # namespace plask::optical::slab

#endif // PLASK__SOLVER__SLAB_EXPANSIONCYL_H
