#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_FINI_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_FINI_H

#include <plask/plask.hpp>

#include "../meshadapter.h"
#include "../patterson.h"
#include "expansioncyl.h"

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

    cvector getHz(const cvector& Bz) override {
        cvector Hz(Bz.size());
        mult_matrix_by_vector(mu_integrals.V_k, Bz, Hz);
        return Hz;
    }

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
