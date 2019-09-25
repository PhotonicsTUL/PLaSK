#ifndef PLASK__SOLVER__SLAB_OLDEXPANSIONCYL_FINI_H
#define PLASK__SOLVER__SLAB_OLDEXPANSIONCYL_FINI_H

#include <plask/plask.hpp>

#include "expansioncyl.h"
#include "../patterson.h"
#include "../meshadapter.h"


namespace plask { namespace optical { namespace slab {

struct PLASK_SOLVER_API ExpansionOldBesselFini: public ExpansionOldBessel {

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionOldBesselFini(OldBesselSolverCyl* solver);

    /// Fill kpts with Bessel zeros
    void computeBesselZeros();

    /// Perform m-specific initialization
    void init2() override;

    /// Free allocated memory
    void reset() override;

    void getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) override;

    double integratePoyntingVert(const cvector& E, const cvector& H) override;

    double integrateField(WhichField field, size_t l, const cvector& E, const cvector& H) override;

  protected:

    /// Integrals for magnetic permeability
    Integrals mu_integrals;

    void layerIntegrals(size_t layer, double lam, double glam) override;

#ifndef NDEBUG
  public:
    cmatrix muVmm();
    cmatrix muVpp();
    cmatrix muTmm();
    cmatrix muTpp();
    cmatrix muTmp();
    cmatrix muTpm();
    cmatrix muDm();
    cmatrix muDp();
    dmatrix muVV();
#endif

};

}}} // # namespace plask::optical::slab

#endif // PLASK__SOLVER__SLAB_OLDEXPANSIONCYL_FINI_H
