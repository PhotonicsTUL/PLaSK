#ifndef PLASK__SOLVER_SLAB_IMPEDANCE_H
#define PLASK__SOLVER_SLAB_IMPEDANCE_H

#include "matrices.hpp"
#include "xance.hpp"
#include "solver.hpp"


namespace plask { namespace optical { namespace slab {

/**
 * Base class for all solvers using reflection matrix method.
 */
struct PLASK_SOLVER_API ImpedanceTransfer: public XanceTransfer {
  
    cvector getReflectionVector(const cvector& incident, IncidentDirection side) override;

    ImpedanceTransfer(SlabBase* solver, Expansion& expansion);

  protected:

    void getFinalMatrix() override;

    void determineFields() override;

    // cvector getReflectionVectorH(const cvector& incident, IncidentDirection side);

    void determineReflectedFields(const cvector& incident, IncidentDirection side) override;

    /**
     * Find impedance matrix for the part of the structure
     * \param start starting layer
     * \param end last layer (reflection matrix is computed for this layer)
     */
    void findImpedance(std::ptrdiff_t start, std::ptrdiff_t end);
};


}}} // namespace plask::optical::slab

#endif // PLASK__SOLVER_SLAB_IMPEDANCE_H
