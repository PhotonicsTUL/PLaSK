#ifndef PLASK__VEC_H
#define PLASK__VEC_H

#include "vector/2d.h"
#include "vector/3d.h"

namespace plask {


/**
* Calculate square of vector magnitude.
* @param v a vector
* @return square of vector magnitude
*/
template <int dim, typename T>
inline double abs2(const Vec<dim,T>& v) { return dot(v,v); }

#ifndef DOXYGEN

template <>
inline double abs2<2,dcomplex>(const Vec<2,dcomplex>& v) { return dot(v,v).real(); }

template <>
inline double abs2<3,dcomplex>(const Vec<3,dcomplex>& v) { return dot(v,v).real(); }

#endif // DOXYGEN

/**
 * Multiple vector @p v by scalar @p scale.
 * @param scale scalar
 * @param v vector
 * @return vector @p v multiplied by @p scalar
 */
template <int dim, typename T, typename OtherT>
inline auto operator*(const OtherT& scale, const Vec<dim,T>& v) -> decltype(v*scale) {
    return v * scale;
}


/**
* Calculate vector magnitude.
* @param v a vector
* @return vector magnitude
*/
template <int dim, typename T>
inline double abs(const Vec<dim,T>& v) { return sqrt(abs2<dim,T>(v)); }

}

#endif // PLASK__VEC_H