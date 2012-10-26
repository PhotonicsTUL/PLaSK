#ifndef PLASK__VEC_H
#define PLASK__VEC_H

#include "vector/2d.h"
#include "vector/3d.h"

namespace plask {


/** @relates Vec
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

/** @relates Vec
 * Multiple vector @p v by scalar @p scale.
 * @param scale scalar
 * @param v vector
 * @return vector @p v multiplied by @p scalar
 */
template <int dim, typename T, typename OtherT>
inline auto operator*(const OtherT& scale, const Vec<dim,T>& v) -> decltype(v*scale) {
    return v * scale;
}


/** @relates Vec
* Calculate vector magnitude.
* @param v a vector
* @return vector magnitude
*/
template <int dim, typename T>
inline double abs(const Vec<dim,T>& v) { return sqrt(abs2<dim,T>(v)); }

namespace details {

    //construct vector in dim space
    template <int dim, typename T>
    struct VecDimConverter {};

    template <typename T>
    struct VecDimConverter<2, T> {
        static Vec<2, T>& get(const Vec<2, T>& src) { return src; }
        static Vec<2, T> get(const Vec<3, T>& src) { return Vec<2, T>(src.tran(), src.vert()); }
    };

    template <typename T>
    struct VecDimConverter<3, T> {
        static Vec<3, T> get(const Vec<2, T>& src) { return Vec<3, T>(T(0.0), src.tran(), src.vert()); }
        static Vec<3, T> get(const Vec<3, T>& src) { return src; }
    };
}


/** @relates Vec
 * Convert vector between space.
 */
template <int dst_dim, typename T, int src_dim>
inline Vec<dst_dim, T> vec(const Vec<src_dim, T>& src) {
    return details::VecDimConverter<dst_dim, T>::get(src);
}



}

#endif // PLASK__VEC_H
