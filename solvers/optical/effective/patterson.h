#ifndef PLASK__PATTERSON_H
#define PLASK__PATTERSON_H

#include <plask/plask.hpp>

/**
 * Patterson quadrature for complex function along specified line
 */

namespace plask { namespace solvers { namespace effective {

extern const double patterson_points[];
extern const double patterson_weights[][256];

/**
 * Compute Patterson quadrature along line a-b in complex plane with specified precision
 * \param fun function to integrate
 * \param a starting point
 * \param b final point
 * \param[in,out] err on input maximum error, on output estimated error
 * \return computed integral
 **/
template <typename R, typename T>
R patterson(const std::function<R(T)>& fun, T a, T b, double& err);


}}} // namespace plask::solvers::effective

#endif // PLASK__PATTERSON_H