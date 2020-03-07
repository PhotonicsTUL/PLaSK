#ifndef PLASK__SOLVER_SLAB_ADMITTANCE_H
#define PLASK__SOLVER_SLAB_ADMITTANCE_H

#include "matrices.h"
#include "xance.h"
#include "solver.h"


namespace plask { namespace optical { namespace slab {

/**
 * Base class for all solvers using reflection matrix method.
 */
struct PLASK_SOLVER_API AdmittanceTransfer: public XanceTransfer {
  
    cvector getReflectionVector(const cvector& incident, IncidentDirection side) override;

    AdmittanceTransfer(SlabBase* solver, Expansion& expansion);

  protected:

    void getFinalMatrix() override;

    void determineFields() override;

    void determineReflectedFields(const cvector& incident, IncidentDirection side) override;

    /**
     * Find admittance matrix for the part of the structure
     * \param start starting layer
     * \param end last layer (reflection matrix is computed for this layer)
     */
    void findAdmittance(std::ptrdiff_t start, std::ptrdiff_t end);
};


}}} // namespace plask::optical::slab

#endif // PLASK__SOLVER_SLAB_ADMITTANCE_H
