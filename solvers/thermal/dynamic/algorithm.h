#ifndef ALGORITHM_H
#define ALGORITHM_H


namespace plask { namespace thermal { namespace dynamic {

/// Choice of matrix factorization algorithms
enum Algorithm {
    ALGORITHM_CHOLESKY, ///< Cholesky factorization
    ALGORITHM_GAUSS    ///< Gauss elimination of asymmetrix matrix (slower but safer as it uses pivoting)
//    ALGORITHM_ITERATIVE ///< Conjugate gradient iterative solver
};

}}}


#endif // ALGORITHM_H
