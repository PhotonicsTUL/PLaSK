#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H

#include <plask/plask.hpp>

#include "expansioncyl.h"
#include "../patterson.h"
#include "../meshadapter.h"


namespace plask { namespace optical { namespace slab {

struct PLASK_SOLVER_API ExpansionBesselInfini: public ExpansionBessel {

    DataVector<double> kdelts;

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionBesselInfini(BesselSolverCyl* solver);

    /// Perform m-specific initialization
    void init2() override;

    void getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) override;

    double integratePoyntingVert(const cvector& E, const cvector& H) override;

    double integrateField(WhichField field, size_t l, const cvector& E, const cvector& H) override;
};

}}} // # namespace plask::optical::slab

#endif // PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
