#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H

#include <plask/plask.hpp>

#include "expansioncyl.h"
#include "../patterson.h"
#include "../meshadapter.h"


namespace plask { namespace optical { namespace slab {

struct PLASK_SOLVER_API ExpansionBesselInfini: public ExpansionBessel {

  protected:

    /// Reference epsilons
    std::vector<std::pair<dcomplex, dcomplex>> eps0;

    DataVector<double> kdelts;

  public:

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionBesselInfini(BesselSolverCyl* solver);

    /// Perform m-specific initialization
    void init2() override;

    /// Free allocated memory
    void reset() override;

    void getMatrices(size_t layer, cmatrix& RE, cmatrix& RH, cmatrix&) override;

    double integratePoyntingVert(const cvector& E, const cvector& H) override;

    double integrateField(WhichField field, size_t l, const cvector& E, const cvector& H) override;

  protected:

    void layerIntegrals(size_t layer, double lam, double glam) override;
};

}}} // # namespace plask::optical::slab

#endif // PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
