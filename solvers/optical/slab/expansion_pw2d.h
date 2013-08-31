#ifndef PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
#define PLASK__SOLVER_SLAB_EXPANSION_PW2D_H

#include <plask/plask.hpp>

#include "expansion.h"
#include "reflection_solver_2d.h"

namespace plask { namespace solvers { namespace slab {

struct ExpansionPW2D: public Expansion {

    /// Solver which performs calculations (and is the interface to the outside world)
    FourierReflection2D* solver;

    /// Indicates of the expansion is a symmetric one
    bool symmetric;

    /**
     * Create new expansion
     * \param solver solver which performs calculations
     */
    ExpansionPW2D(FourierReflection2D* solver);

    virtual size_t lcount() const;

    virtual bool diagonalQE(size_t l) const;

    virtual size_t matrixSize() const;

    virtual cmatrix getRE(size_t l, dcomplex k0, dcomplex kx, dcomplex ky);

    virtual cmatrix getRH(size_t l, dcomplex k0, dcomplex kx, dcomplex ky);

  protected:

    /**
     * Compute expansion coefficients for material parameters
     * \param l layer number
     */
    DataVector<const Tensor3<dcomplex>> getMaterialParameters(size_t l);

};

}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
