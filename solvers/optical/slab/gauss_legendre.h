#ifndef PLASK__SOLVER__SLAB_GAUSS_LEGENDRE_H
#define PLASK__SOLVER__SLAB_GAUSS_LEGENDRE_H

#include <plask/plask.hpp>

namespace plask { namespace optical { namespace slab {

/**
 * Compute ascissae and weights for Gauss-Legendre quadatures.
 * \param n quadrature order
 * \param[out] abscissae computed abscissae
 * \param[out] weights corresponding weights
 */
void gaussLegendre(size_t n, std::vector<double>& abscissae, DataVector<double>& weights);

}}} // # namespace plask::optical::slab

#endif // PLASK__SOLVER__SLAB_GAUSS_LEGENDRE_H
