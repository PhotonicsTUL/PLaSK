#ifndef PLASK__SOLVER__ELECTRICAL_SHOCKLEY_COMMON_H
#define PLASK__SOLVER__ELECTRICAL_SHOCKLEY_COMMON_H

#include <plask/plask.hpp>

#include "block_matrix.h"
#include "gauss_matrix.h"
#include "conjugate_gradient.h"

namespace plask { namespace electrical { namespace shockley {
    
/// Choice of matrix factorization algorithms
enum Algorithm {
    ALGORITHM_CHOLESKY, ///< Cholesky factorization
    ALGORITHM_GAUSS,    ///< Gauss elimination of asymmetrix matrix (slower but safer as it uses pivoting)
    ALGORITHM_ITERATIVE ///< Conjugate gradient iterative solver
};

/// Choice of heat computation method in active region
enum HeatMethod {
    HEAT_JOULES, ///< compute Joules heat using effective conductivity
    HEAT_BANDGAP ///< compute heat based on the size of the band gap
};

}}} // # namespace plask::electrical::shockley

#endif // PLASK__SOLVER__ELECTRICAL_SHOCKLEY_COMMON_H
