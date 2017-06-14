#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_FINI_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_FINI_H

#include <plask/plask.hpp>

#include "expansioncyl.h"
#include "../patterson.h"
#include "../meshadapter.h"


namespace plask { namespace solvers { namespace slab {

struct PLASK_SOLVER_API ExpansionBesselFini: public ExpansionBessel {

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionBesselFini(BesselSolverCyl* solver);

    /// Fill factors with Bessel zeros
    void computeBesselZeros();

    /// Perform m-specific initialization
    void init2() override;

    /// Free allocated memory
    void reset() override;

    void getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) override;

  protected:

    /// Integrals for magnetic permeability
    Integrals mu_integrals;

  public:

#ifndef NDEBUG
    cmatrix muVmm();
    cmatrix muVpp();
    cmatrix muTmm();
    cmatrix muTpp();
    cmatrix muTmp();
    cmatrix muTpm();
    cmatrix muDm();
    cmatrix muDp();
#endif

};

}}} // # namespace plask::solvers::slab

#endif // PLASK__SOLVER__SLAB_EXPANSIONCYL_FINI_H
