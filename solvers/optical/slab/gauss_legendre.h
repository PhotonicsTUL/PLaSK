#ifndef PLASK__SOLVER__SLAB_GAUSS_LEGENDRE_H
#define PLASK__SOLVER__SLAB_GAUSS_LEGENDRE_H

#include <plask/plask.hpp>

namespace plask { namespace solvers { namespace slab {
    
/**
 * Compute ascissae and weights for Gauss-Legendre quadatures.
 * \param n quadrature order
 * \param[out] abscissae computed abscissae
 * \param[out] weights corresponding weights
 */
void gaussData(size_t n, std::vector<double>& abscissae, DataVector<double>& weights);
    
}}} // # namespace plask::solvers::slab

#endif // PLASK__SOLVER__SLAB_GAUSS_LEGENDRE_H