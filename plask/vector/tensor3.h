#ifndef PLASK__TESNOR3_H
#define PLASK__TESNOR3_H

/** @file
This file contains implementation of tensor in 2D space.
*/

#include <iostream>

#include "../math.h"

namespace plask {

/**
 * Non-diagonal tensor with all non-diagonal lateral projection.
 * [ c00 c01  0  ]
 * [ c01 c11  0  ]
 * [  0   0  c22 ]
 */
template <typename T>
struct Tensor3 {

    T c00, ///< Value of the tensor in LONG direction
      c11, ///< Value of the tensor in TRAN direction
      c22, ///< Value of the tensor in VERT direction
      c01; ///< Non-diagonal component LONG-TRAN

    T& tran() { return c00; }
    const T& lon() const { return c00; }

    T& lon() { return c11; }
    const T& tran() const { return c11; }

    T& vert() { return c22; }
    const T& vert() const { return c22; }

    /// Construct uninitialized Tensor.
    Tensor3() {}

    /**
     * Copy constructor from all other 3d tensors.
     * @param p tensor to copy from
     */
    template <typename OtherT>
    Tensor3(const Tensor3<OtherT>& p): c00(p.c00), c11(p.c11), c22(p.c22), c01(p.c01) {}

    /**
     * Construct isotropic tensor.
     * @param val value
     */
    Tensor3(const T& val): c00(val), c11(val), c22(val), c01(0.) {}

    /**
     * Construct tensor with given diagonal values.
     * @param c00, c22 components
     */
    Tensor3(const T& c00, const T& c22): c00(c00), c11(c00), c22(c22), c01(0.) {}

    /**
     * Construct tesors with given diagonal values.
     * @param c00, c11, c22 components
     */
    Tensor3(const T& c00, const T& c11, const T& c22): c00(c00), c11(c11), c22(c22), c01(0.) {}

    /**
     * Construct tesors with given all values.
     * @param c00, c11, c22, c01 components
     */
    Tensor3(const T& c00, const T& c11, const T& c22, const T& c01): c00(c00), c11(c11), c22(c22), c01(c01) {}

    /**
     * Construct tensor components given in std::pair.
     * @param comp components
     */
    template <typename T0, typename T1>
    Tensor3(const std::pair<T0,T1>& comp): c00(comp.first), c11(comp.first), c22(comp.second), c01(0.) {}

    /// Convert to std::tuple
    operator std::tuple<T,T,T,T,T>() const {
        return std::make_tuple(c00, c11, c22, c01);
    }

    /**
     * Compare two tensors, this and @p p.
     * @param p tensor to compare
     * @return true only if this tensor and @p p have equals coordinates
     */
    template <typename OtherT>
    bool operator==(const Tensor3<OtherT>& p) const {
        return p.c00 == c00 && p.c11 == c11 && p.c22 == c22 && p.c01 == c01;
    }

    /**
     * Compare two tensors, this and @p p.
     * @param p tensor to compare
     * @return true only if this tensor and @p p don't have equals coordinates
     */
    template <typename OtherT>
    bool operator!=(const Tensor3<OtherT>& p) const {
        return p.c00 != c00 || p.c11 != c11 || p.c22 != c22 || p.c01 != c01;
    }

    /**
     * Calculate sum of two tesnors, @c this and @p other.
     * @param other tensor to add, can have different data type (than result type will be found using C++ types promotions rules)
     * @return tensors sum
     */
    template <typename OtherT>
    auto operator+(const Tensor3<OtherT>& other) const -> Tensor3<decltype(c00 + other.c00)> {
        return Tensor3<decltype(this->c00 + other.c00)>
            (c00 + other.c00, c11 + other.c11, c22 + other.c22, c01 + other.c01);
    }

    /**
     * Increase coordinates of this tensor by coordinates of other tensor @p other.
     * @param other tensor to add
     * @return *this (after increase)
     */
    Tensor3<T>& operator+=(const Tensor3<T>& other) {
        c00 += other.c00;
        c11 += other.c11;
        c22 += other.c22;
        c01 += other.c01;
        return *this;
    }

    /**
     * Calculate difference of two tensors, @c this and @p other.
     * @param other tensor to subtract from this, can have different data type (than result type will be found using C++ types promotions rules)
     * @return tensors difference
     */
    template <typename OtherT>
    auto operator-(const Tensor3<OtherT>& other) const -> Tensor3<decltype(c00 - other.c00)> {
        return Tensor3<decltype(this->c00 - other.c00)>
            (c00 - other.c00, c11 - other.c11, c22 - other.c22, c01 - other.c01);
    }

    /**
     * Decrease coordinates of this tensor by coordinates of other tensor @p other.
     * @param other tensor to subtract
     * @return *this (after decrease)
     */
    Tensor3<T>& operator-=(const Tensor3<T>& other) {
        c00 -= other.c00;
        c11 -= other.c11;
        c22 -= other.c22;
        c01 -= other.c01;
        return *this;
    }

    /**
     * Calculate this tensor multiplied by scalar @p scale.
     * @param scale scalar
     * @return this tensor multiplied by scalar
     */
    template <typename OtherT>
    auto operator*(const OtherT scale) const -> Tensor3<decltype(c00*scale)> {
        return Tensor3<decltype(c00*scale)>(c00 * scale, c11 * scale, c22 * scale, c01 * scale);
    }

    /**
     * Multiple coordinates of this tensor by @p scalar.
     * @param scalar scalar
     * @return *this (after scale)
     */
    Tensor3<T>& operator*=(const T scalar) {
        c00 *= scalar;
        c11 *= scalar;
        c22 *= scalar;
        c01 *= scalar;
        return *this;
    }

    /**
     * Calculate this tensor divided by scalar @p scale.
     * @param scale scalar
     * @return this tensor divided by scalar
     */
    Tensor3<T> operator/(const T scale) const {
        return Tensor3<decltype(c00/scale)>(c00 / scale, c11 / scale, c22 / scale, c01 / scale);
    }

    /**
     * Divide coordinates of this tensor by @p scalar.
     * @param scalar scalar
     * @return *this (after divide)
     */
    Tensor3<T>& operator/=(const T scalar) {
        c00 /= scalar;
        c11 /= scalar;
        c22 /= scalar;
        c01 /= scalar;
        return *this;
    }

    /**
     * Calculate tensor opposite to this.
     * @return Tensor3<T>(-c00, -c11)
     */
    Tensor3<T> operator-() const {
        return Tensor3<T>(-c00, -c11, -c22, -c01);
    }

    /**
     * Square each component of tensor
     * \return squared tensor
     */
    Tensor3<T> sqr() const {
        return Tensor3<T>(c00*c00, c11*c11, c22*c22, c01*c01);
    }

    /**
     * Square each component of tensor in place
     * \return *this (squared)
     */
    Tensor3<T>& sqr_inplace() {
        c00 *= c00; c11 *= c11; c22 *= c22; c01 *= c01;
        return *this;
    }

    /**
     * Square root each component of tensor
     * \return squared tensor
     */
    Tensor3<T> sqrt() const {
        return Tensor3<T>(std::sqrt(c00), std::sqrt(c11), std::sqrt(c22), std::sqrt(c01));
    }

    /**
     * Square root each component of tensor in place
     * \return *this (squared)
     */
    Tensor3<T>& sqrt_inplace() {
        c00 = std::sqrt(c00); c11 = std::sqrt(c11); c22 = std::sqrt(c22); c01 = std::sqrt(c01);
        return *this;
    }

    /**
     * Inverse of the tensor
     * \return inverse of this
     */
    Tensor3<T> inv() const {
        T M = c00*c11 - c01*c01;
        return Tensor3<dcomplex>(c11/M, c00/M, 1./c22, -c01/M);
    }

    /**
     * Print tensor to stream using format (where c00 and c11 are tensor components): [c00, c11]
     * @param out print destination, output stream
     * @param to_print tensor to print
     * @return out stream
     */
    friend inline std::ostream& operator<<(std::ostream& out, const Tensor3<T>& to_print) {
        return out << '(' << to_print.c00 << ", " <<  to_print.c11 << ", " << to_print.c22 << ", ("
                          << to_print.c01 << "))";
    }
};

/**
 * Calculate this tensor multiplied by scalar \p scale.
 * \param scale scalar
 * \param tensor tensor
 * \return this tensor multiplied by scalar
 */
template <typename T, typename OtherT>
auto operator*(const OtherT scale, const Tensor3<T>& tensor) -> decltype(tensor*scale) {
    return tensor * scale;
}

/**
 * Calculate tensor conjugate.
 * @param v a tensor
 * @return conjugate tensor
 */
template <typename T>
inline Tensor3<T> conj(const Tensor3<T>& v) { return Tensor3<T>(conj(v.c00), conj(v.c11), conj(v.c22), conj(v.c01)); }

} //namespace plask

#endif // PLASK__TESNOR3_H
