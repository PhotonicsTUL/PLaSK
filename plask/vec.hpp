/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#ifndef PLASK__VEC_H
#define PLASK__VEC_H

#include "vector/2d.hpp"
#include "vector/3d.hpp"

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
 * Multiply vector @p v by scalar @p scale.
 * @param scale scalar
 * @param v vector
 * @return vector @p v multiplied by @p scalar
 */
template <int dim, typename T, typename OtherT>
inline constexpr auto operator*(const OtherT& scale, const Vec<dim,T>& v) -> decltype(v*scale) {
    return v * scale;
}


/** @relates Vec
* Calculate vector magnitude.
* @param v a vector
* @return vector magnitude
*/
template <int dim, typename T>
inline double abs(const Vec<dim,T>& v) { return sqrt(abs2<dim,T>(v)); }

template <>
inline double abs<2, double>(const Vec<2,double>& v) { return std::hypot(v.c0, v.c1); }

/** @relates Vec
* Check if any vector componen is not-a-number.
* @param v a vector
* @return \c true if any vector magnitude is Nan
*/
template <int dim, typename T>
bool isnan(const Vec<dim,T>& v) {
    for (int i = 0; i < dim; ++i) if (isnan(v[i])) return true;
    return false;
}

namespace details {

    //construct vector in dim space
    template <int dim, typename T>
    struct VecDimConverter {};

    template <typename T>
    struct VecDimConverter<2, T> {
        static Vec<2,T>& get(const Vec<2,T>& src) { return src; }
        static Vec<2,T> get(const Vec<3,T>& src) { return Vec<2, T>(src.tran(), src.vert()); }
    };

    template <typename T>
    struct VecDimConverter<3, T> {
        static Vec<3,T> get(const Vec<2,T>& src) { return Vec<3, T>(T(0.0), src.tran(), src.vert()); }
        static Vec<3,T> get(const Vec<3,T>& src) { return src; }
    };
}


/** @relates Vec
 * Convert vector between space.
 * @param src source vector
 * @return @p src vector in destination space
 */
template <int dst_dim, typename T, int src_dim>
inline Vec<dst_dim, T> vec(const Vec<src_dim, T>& src) {
    return details::VecDimConverter<dst_dim, T>::get(src);
}

/** @relates Vec
 * Convert vector 2D to 3D space.
 * @param src source vector
 * @param lon longitude coordinate
 * @return @p src vector in destination space and given @p lon coordinate
 */
template <typename T>
inline Vec<3, T> vec(const Vec<2, T>& src, T lon) {
    return Vec<3, T>(lon, src.tran(), src.vert());
}

/** @relates Vec
 * Rotate @p v over up axis to lie on tran-vert plane.
 * @param v vector in 3D space
 * @return vector in 2D space, after rotation, with tran()>=0 and vert()==v.vert()
 */
inline Vec<2, double> rotateToLonTranAbs(const Vec<3, double>& v) {
    return vec(std::hypot(v.lon(), v.tran()), v.vert());
}

/** @relates Vec
 * Rotate @p v over up axis to lie on tran-vert plane.
 * @param v vector in 3D space
 * @return vector in 2D space, after rotation, with sign of tran() same as v.tran() and vert()==v.vert()
 */
inline Vec<2, double> rotateToLonTranSgn(const Vec<3, double>& v) {
    return vec(std::copysign(std::hypot(v.lon(), v.tran()), v.tran()), v.vert());
}

/**
 * @relates Vec
 * Calculate @p v3 + @p v2 with given @p lon coordinate.
 * @param v3 vector in 3D space
 * @param v2 vector in 2D space
 * @param lon last coordinate of v2
 * @return @p v3 + @p v2 with given @p lon coordinate
 */
template <typename T>
inline Vec<3, T> vec3Dplus2D(const Vec<3, T>& v3, const Vec<2, T>& v2, T lon) {
    return Vec<3, T>(v3.lon() + lon, v3.tran() + v2.tran(), v3.vert() + v2.vert());
}

/**
 * @relates Vec
 * Calculate @p v3 + @p v2, same as vec3Dplus2D(v3, v2, 0.0).
 * @param v3 vector in 3D space
 * @param v2 vector in 2D space
 * @return @p v3 + @p v2
 */
template <typename T>
inline Vec<3, T> vec3Dplus2D(const Vec<3, T>& v3, const Vec<2, T>& v2) {
    return Vec<3, T>(v3.lon(), v3.tran() + v2.tran(), v3.vert() + v2.vert());
}

}

#endif // PLASK__VEC_H
