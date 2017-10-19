#ifndef PLASK__PATTERSON_H
#define PLASK__PATTERSON_H

#include <plask/plask.hpp>

#include "patterson-data.h"

/**
 * Patterson quadrature for complex function along specified line
 */

namespace plask { namespace optical { namespace effective {

/**
 * Compute Patterson quadrature along line a-b in complex plane with specified precision
 * \param fun function to integrate
 * \param a starting point
 * \param b final point
 * \param[in,out] err on input maximum error, on output estimated error
 * \return computed integral
 **/
template <typename S, typename T>
S patterson(const std::function<S(T)>& fun, T a, T b, double& err);


}}} // namespace plask::optical::effective

#endif // PLASK__PATTERSON_H