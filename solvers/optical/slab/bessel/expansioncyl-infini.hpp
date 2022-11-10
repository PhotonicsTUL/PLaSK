#ifndef PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
#define PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H

#include <plask/plask.hpp>

#include "expansioncyl.hpp"
#include "../patterson.hpp"
#include "../meshadapter.hpp"


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

    virtual void integrateParams(Integrals& integrals,
                                 const dcomplex* datap, const dcomplex* datar, const dcomplex* dataz,
                                 dcomplex datap0, dcomplex datar0, dcomplex dataz0) override;

    double fieldFactor(size_t i) override {
        return rbounds[rbounds.size() - 1] / (kpts[i] * kdelts[i]);
    }

    cmatrix getHzMatrix(const cmatrix& Bz, cmatrix& Hz) override { return Bz; }
};

}}} // # namespace plask::optical::slab

#endif // PLASK__SOLVER__SLAB_EXPANSIONCYL_INFINI_H
