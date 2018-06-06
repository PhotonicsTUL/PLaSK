#ifndef PLASK__SOLVER_SEMIVECTORIALBASE_H
#define PLASK__SOLVER_SEMIVECTORIALBASE_H

#include <plask/plask.hpp>
#include "rootdigger.h"

namespace plask { namespace optical { namespace semivectorial {

struct PLASK_SOLVER_API SemiVectorialBase {

public:
    
    /// Layer boundraries
    shared_ptr<OrderedAxis> vbounds;
    
    /// Centers of layers
    shared_ptr<OrderedAxis> verts;
    
    /// Number of distinct layers
    size_t lcount;
    
    /// Organization of layers in the stack
    std::vector<std::size_t> stack;
    
    /// Reference wavelength used for getting material parameters [nm]
    double lam0;
    
    /// Normalized frequency [1/um]
    dcomplex k0;
    
    /// Parameters for main rootdigger
    RootDigger::Params root;
    
    SemiVectorialBase():
        lam0(NAN),
        k0(NAN)
        {}
    
};

}}}
#endif // PLASK__SOLVER_SEMIVECTORIABASE_H