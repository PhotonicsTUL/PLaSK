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

  protected:

    double fieldFactor(size_t i) override {
        return rbounds[rbounds.size() - 1] / (kpts[i] * kdelts[i]);
    }

    cvector getHz(const cvector& Bz) override { return Bz; }
};

}}} // # namespace plask::optical::slab

#endif // PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
