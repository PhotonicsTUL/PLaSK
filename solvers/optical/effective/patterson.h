#ifndef PLASK__PATTERSON_H
#define PLASK__PATTERSON_H

#include <plask/plask.hpp>

/**
 * Patterson quadrature for complex function along specified line
 */

namespace plask { namespace solvers { namespace effective {

/**
 * Compute Patterson quadrature along line a-b in complex plane with specified precision
 * \param fun function to integrate
 * \param a starting point
 * \param b final point
 * \return computed integral
 **/
dcomplex patterson(const std::function<dcomplex(dcomplex)>& fun, dcomplex a, dcomplex b, double eps=1e-15);


}}} // namespace plask::solvers::effective

#endif // PLASK__PATTERSON_H