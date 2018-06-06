#ifndef PLASK__SOLVER_SEMIVECTORIAL_EXPANSION_H
#define PLASK__SOLVER_SEMIVECTORIAL_EXPANSION_H

#include <plask/plask.hpp>
#include "semivectorial.h"
#include "solver.h"

namespace plask { namespace optical { namespace semivectorial {

struct PLASK_SOLVER_API Expansion {
       
    /// Specified component in polarization or symmetry
    enum Component {
        E_UNSPECIFIED = 0, ///< All components exist or no symmetry
        E_TRAN = 1,         ///< E_tran and H_long exist or are symmetric and E_long and H_tran anti-symmetric
        E_LONG = 2          ///< E_long and H_tran exist or are symmetric and E_tran and H_long anti-symmetric
    };
    
    enum WhichField { 
        FIELD_E,
        FIELD_H 
    };
    
    WhichField which_field;
    
    /// Solver which performs calculations
    SemiVectorialBase* solver;
    
    /// Frequency for which the actual computations are performed
    dcomplex k0;
    
    /// Material parameters wavelength
    double lam0;
    
    Expansion(SemiVectorialBase* solver): solver(solver), k0(NAN), lam0(NAN) {}
    
    
};
}}}

#endif //PLASK__SOLVER_SEMIVECTORIAL_EXPANSION_H