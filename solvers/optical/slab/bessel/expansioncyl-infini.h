#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H

#include <plask/plask.hpp>

#include "expansioncyl.h"
#include "../patterson.h"
#include "../meshadapter.h"


namespace plask { namespace solvers { namespace slab {

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

    void getMatrices(size_t layer, cmatrix& RE, cmatrix& RH) override;

  protected:

    void layerIntegrals(size_t layer, double lam, double glam) override;
};

}}} // # namespace plask::solvers::slab

#endif // PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
