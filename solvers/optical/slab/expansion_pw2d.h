#ifndef PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
#define PLASK__SOLVER_SLAB_EXPANSION_PW2D_H

#include <plask/plask.hpp>

#include "expansion.h"

namespace plask { namespace solvers { namespace slab {
    
struct ExpansionPW2D: public Expansion {

    virtual size_t lcount() const;

    virtual bool diagonalQE(size_t l) const;
    
    virtual size_t matrixSize() const;
    
    virtual cmatrix getRE(size_t l, dcomplex k0, dcomplex kx, dcomplex ky);
    
    virtual cmatrix getRH(size_t l, dcomplex k0, dcomplex kx, dcomplex ky);

    
};    
    
}}} // namespace plask

#endif // PLASK__SOLVER_SLAB_EXPANSION_PW2D_H
