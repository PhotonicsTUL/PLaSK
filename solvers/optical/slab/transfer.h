#ifndef PLASK__SLAB_TRANSFER_H
#define PLASK__SLAB_TRANSFER_H

#include <plask/solver.h>

#include "diagonalizer.h"

namespace plask { namespace solvers { namespace slab {

struct SlabBase;

/**
 * Base class for Admittance and Reflection transfers.
 */
struct PLASK_SOLVER_API Transfer {

    /// Direction specification for reflection calculations
    enum IncidentDirection {
        INCIDENCE_TOP,      ///< Incident light propagating from top (downwards)
        INCIDENCE_BOTTOM    ///< Incident light propagating from bottom (upwards)
    };

    /// Indicates what has been determined
    enum Determined {
        DETERMINED_NOTHING = 0, ///< Nothing has been determined
        DETERMINED_RESONANT,    ///< Resonant field has been determined
        DETERMINED_REFLECTED    ///< Reflected field has been determined
    };

    /// Solver containing this transfer
    SlabBase* solver;

    /// Diagonalizer used to compute matrix of eigenvalues and eigenvectors
    std::unique_ptr<Diagonalizer> diagonalizer;

    Determined fields_determined;               ///< Are the diagonalized fields determined for all layers?
    /**
     * Create transfer object and initialize memory
     * \param solver solver counting this transfer
     * \param expansion expansion for diagonalizer
     */
    Transfer(SlabBase* solver, Expansion& expansion):
        solver(solver),
        diagonalizer(new SimpleDiagonalizer(&expansion)),   //TODO add other diagonalizer types
        fields_determined(DETERMINED_NOTHING)
    {}

    /// Compute discontinuity matrix determinant for the current parameters
    virtual dcomplex determinant() = 0;
};


}}} // namespace plask::solvers::slab

#endif // PLASK__SLAB_TRANSFER_H
