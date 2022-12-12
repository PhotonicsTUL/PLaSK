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
#ifndef PLASK__TESNOR2_H
#define PLASK__TESNOR2_H

/** @file
This file contains implementation of tensor in 2D space.
*/

#include <iostream>
#include <boost/concept_check.hpp>

#include "../math.hpp"
#include "2d.hpp"

namespace plask {

/**
 * Diagonal tensor with all lateral components equal.
 * [ c00  0  ]
 * [  0  c11 ]
 */
template <typename T>
struct Tensor2 {

    T c00, ///< Value of the tensor in lateral direction
      c11; ///< Value of the tensor in vertical direction

    T& tran() { return c00; }
    const T& tran() const { return c00; }

    T& vert() { return c11; }
    const T& vert() const { return c11; }

    /// Construct uninitialized Tensor.
    Tensor2() {}

    /**
     * Copy constructor from all other 2d tensors.
     * @param p tensor to copy from
     */
    template <typename OtherT>
    Tensor2(const Tensor2<OtherT>& p): c00(p.c00), c11(p.c11) {}

    /**
     * Construct isotropic tensor.
     * @param val value
     */
    Tensor2(const T& val): c00(val), c11(val) {}

    /**
     * Construct tensor with given diagonal values.
     * @param c00, c11 components
     */
    Tensor2(const T& c00, const T& c11): c00(c00), c11(c11) {}

    /**
     * Construct tensor components given in std::pair.
     * @param comp components
     */
    template <typename T0, typename T1>
    Tensor2(const std::pair<T0,T1>& comp): c00(comp.first), c11(comp.second) {}

    /**
     * Construct tensor from 2D vector
     * \param vec vector
     */
    Tensor2(const Vec<2,T>& vec): c00(vec.c0), c11(vec.c1) {}

   /**
     * Get i-th component
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * \param i number of coordinate
     * \return i-th component
     */
    inline T& operator[](size_t i) {
        assert(i < 2);
        return *(&c00 + i);
    }

    /**
     * Get i-th component
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * \param i number of coordinate
     * \return i-th component
     */
    inline const T& operator[](size_t i) const {
        assert(i < 2);
        return *(&c00 + i);
    }

    /// Convert to std::tuple
    operator std::tuple<T,T>() const {
        return std::make_tuple(c00, c11);
    }

    /**
     * Compare two tensors, this and @p p.
     * @param p tensor to compare
     * @return true only if this tensor and @p p have equals coordinates
     */
    template <typename OtherT>
    bool operator==(const Tensor2<OtherT>& p) const { return p.c00 == c00 && p.c11 == c11; }

    /**
     * Check if two tensors, this and @p p are almost equal.
     * @param p tensors to compare
     * @return @c true only if this tensors and @p p have almost equals coordinates
     */
    template <typename OtherT>
    constexpr bool equals(const Tensor2<OtherT>& p) const {
        return is_zero(p.c00 - c00) && is_zero(p.c11 - c11);
    }

    /**
     * Compare two tensors, this and @p p.
     * @param p tensor to compare
     * @return true only if this tensor and @p p don't have equals coordinates
     */
    template <typename OtherT>
    bool operator!=(const Tensor2<OtherT>& p) const { return p.c00 != c00 || p.c11 != c11; }

    /**
     * Calculate sum of two tesnors, @c this and @p other.
     * @param other tensor to add, can have different data type (than result type will be found using C++ types promotions rules)
     * @return tensors sum
     */
    template <typename OtherT>
    auto operator+(const Tensor2<OtherT>& other) const -> Tensor2<decltype(c00 + other.c00)> {
        return Tensor2<decltype(this->c00 + other.c00)>(c00 + other.c00, c11 + other.c11);
    }

    /**
     * Increase coordinates of this tensor by coordinates of other tensor @p other.
     * @param other tensor to add
     * @return *this (after increase)
     */
    Tensor2<T>& operator+=(const Tensor2<T>& other) {
        c00 += other.c00;
        c11 += other.c11;
        return *this;
    }

    /**
     * Calculate difference of two tensors, @c this and @p other.
     * @param other tensor to subtract from this, can have different data type (than result type will be found using C++ types promotions rules)
     * @return tensors difference
     */
    template <typename OtherT>
    auto operator-(const Tensor2<OtherT>& other) const -> Tensor2<decltype(c00 - other.c00)> {
        return Tensor2<decltype(this->c00 - other.c00)>(c00 - other.c00, c11 - other.c11);
    }

    /**
     * Decrease coordinates of this tensor by coordinates of other tensor @p other.
     * @param other tensor to subtract
     * @return *this (after decrease)
     */
    Tensor2<T>& operator-=(const Tensor2<T>& other) {
        c00 -= other.c00;
        c11 -= other.c11;
        return *this;
    }

    /**
     * Calculate this tensor multiplied by scalar @p scale.
     * @param scale scalar
     * @return this tensor multiplied by scalar
     */
    template <typename OtherT>
    auto operator*(const OtherT scale) const -> Tensor2<decltype(c00*scale)> {
        return Tensor2<decltype(c00*scale)>(c00 * scale, c11 * scale);
    }

    /**
     * Multiple coordinates of this tensor by @p scalar.
     * @param scalar scalar
     * @return *this (after scale)
     */
    Tensor2<T>& operator*=(const T scalar) {
        c00 *= scalar;
        c11 *= scalar;
        return *this;
    }

    /**
     * Calculate this tensor divided by scalar @p scale.
     * @param scale scalar
     * @return this tensor divided by scalar
     */
    Tensor2<T> operator/(const T scale) const { return Tensor2<T>(c00 / scale, c11 / scale); }

    /**
     * Divide coordinates of this tensor by @p scalar.
     * @param scalar scalar
     * @return *this (after divide)
     */
    Tensor2<T>& operator/=(const T scalar) {
        c00 /= scalar;
        c11 /= scalar;
        return *this;
    }

    /**
     * Calculate tensor opposite to this.
     * @return Tensor2<T>(-c00, -c11)
     */
    Tensor2<T> operator-() const {
        return Tensor2<T>(-c00, -c11);
    }

    /**
     * Square each component of tensor
     * \return squared tensor
     */
    Tensor2<T> sqr() const {
        return Tensor2<T>(c00*c00, c11*c11);
    }

    /**
     * Square each component of tensor in place
     * \return *this (squared)
     */
    Tensor2<T>& sqr_inplace() {
        c00 *= c00; c11 *= c11;
        return *this;
    }

    /**
     * Square root of each component of tensor
     * \return squared tensor
     */
    Tensor2<T> sqrt() const {
        return Tensor2<T>(std::sqrt(c00), std::sqrt(c11));
    }

    /**
     * Square root of each component of tensor in place
     * \return *this (squared)
     */
    Tensor2<T>& sqrt_inplace() {
        c00 = std::sqrt(c00); c11 = std::sqrt(c11);
        return *this;
    }

    /**
     * Power of each component of tensor
     * \return squared tensor
     */
    template <typename OtherT>
    Tensor2<T> pow(OtherT a) const {
        return Tensor2<T>(std::pow(c00, a), std::pow(c11, a));
    }

    /**
     * Print tensor to stream using format (where c00 and c11 are tensor components): [c00, c11]
     * @param out print destination, output stream
     * @param to_print tensor to print
     * @return out stream
     */
    friend inline std::ostream& operator<<(std::ostream& out, const Tensor2<T>& to_print) {
        return out << '(' << str(to_print.c00) << ", " << str(to_print.c11) << ')';
    }

};

/**
 * Calculate this tensor multiplied by scalar \p scale.
 * \param scale scalar
 * \param tensor tensor
 * \return this tensor multiplied by scalar
 */
template <typename T, typename OtherT>
auto operator*(const OtherT scale, const Tensor2<T>& tensor) -> decltype(tensor*scale) {
    return tensor * scale;
}

/**
 * Calculate tensor conjugate.
 * @param v a tensor
 * @return conjugate tensor
 */
template <typename T>
inline Tensor2<T> conj(const Tensor2<T>& v) { return Tensor2<T>(conj(v.c00), conj(v.c11)); }

/// Specialization of NaNImpl which add support for 2D tensors.
template <typename T>
struct NaNImpl<Tensor2<T>> {
    static constexpr Tensor2<T> get() { return Tensor2<T>(NaN<T>()); }
};

/// Specialization of ZeroImpl which add support for 2D vectors.
template <typename T>
struct ZeroImpl<Tensor2<T>> {
    static constexpr Tensor2<T> get() { return Tensor2<T>(0.); }
};

/// Check if the tensor is almost zero
/// \param v tensor to verify
template <typename T>
inline bool is_zero(const Tensor2<T>& v) {
    return is_zero(v.c00) && is_zero(v.c11);
}

/*
PLASK_API_EXTERN_TEMPLATE_STRUCT(Tensor2<double>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(Tensor2< std::complex<double> >)
*/

} //namespace plask

namespace std {
    template <typename T>
    plask::Tensor2<T> sqrt(plask::Tensor2<T> tens) {
        return tens.sqrt();
    }

    template <typename T, typename OtherT>
    plask::Tensor2<T> pow(plask::Tensor2<T> tens, OtherT a) {
        return tens.pow(a);
    }
}

#endif // PLASK__TESNOR2_H
