#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_H

#include <plask/plask.hpp>

#include "../expansion.h"
#include "../patterson.h"
#include "../meshadapter.h"

namespace plask { namespace solvers { namespace slab {

struct BesselSolverCyl;

struct PLASK_SOLVER_API ExpansionBessel: public Expansion {

    int m;                              ///< Angular dependency index

    bool initialized;                   ///< Expansion is initialized

    bool m_changed;                     ///< m has changed and init2 must be called

    /// Horizontal axis with separate integration intervals.
    /// material functions contain discontinuities at these points
    OrderedAxis rbounds;

    ///  Argument coefficients for Bessel expansion base (zeros of Bessel function for finite domain)
    std::vector<double> factors;

    /// Mesh for getting material data
    shared_ptr<RectangularMesh<2>> mesh;

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionBessel(BesselSolverCyl* solver);

    /// Init expansion
    void init1();

    /// Perform m-specific initialization
    virtual void init2() = 0;

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

    double integratePoyntingVert(const cvector& E, const cvector& H) override;

  protected:

    /// Integration segment data
    struct Segment {
        double Z;                       ///< Center of the segment
        double D;                       ///< Width of the segment divided by 2
        DataVector<double> weights;     ///< Cached integration weights for segment
    };

    /// Integration segments
    std::vector<Segment> segments;

    /// Cached eps^(-1)
    std::vector<DataVector<dcomplex>> iepsilons;

    /// Information if the layer is diagonal
    std::vector<bool> diagonals;

    /// Obtained temperature
    LazyData<double> temperature;

    /// Flag indicating if the gain is connected
    bool gain_connected;

    /// Obtained gain
    LazyData<double> gain;

    void prepareIntegrals(double lam, double glam) override;

    void cleanupIntegrals(double lam, double glam) override;

  public:

    void setM(unsigned n) {
        if (n != m) {
            write_debug("{0}: m changed from {1} to {2}", solver->getId(), m, n);
            m = n;
            solver->recompute_integrals = true;
            solver->clearFields();
        }
    }

    /// Get \f$ X_s \f$ index
    size_t idxs(size_t i) { return 2 * i; }

    /// Get \f$ X_p \f$ index
    size_t idxp(size_t i) { return 2 * i + 1; }

};

}}} // # namespace plask::solvers::slab

#endif // PLASK__SOLVER__SLAB_EXPANSIONCYL_H
