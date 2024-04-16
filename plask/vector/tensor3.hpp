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
#ifndef PLASK__TENSOR3_H
#define PLASK__TENSOR3_H

/** @file
This file contains implementation of tensor in 2D space.
*/

#include <iostream>

#include "../math.hpp"
#include "2d.hpp"
#include "3d.hpp"
#include "tensor2.hpp"

namespace plask {

/**
 * Non-diagonal tensor with all non-diagonal lateral projection.
 * [ c00 c01 c02 ]
 * [ c10 c11 c12 ]
 * [ c20 c21 c22 ]
 */
template <typename T> struct Tensor3 {
    // clang-format off
    T c00,    ///< Value of the tensor in LONG direction
      c01,    ///< Non-diagonal component LONG-TRAN
      c02,    ///< Non-diagonal component LONG-VERT
      c10,    ///< Non-diagonal component TRAN-LONG
      c11,    ///< Value of the tensor in TRAN direction
      c12,    ///< Non-diagonal component TRAN-VERT
      c20,    ///< Non-diagonal component VERT-LONG
      c21,    ///< Non-diagonal component VERT-TRAN
      c22;    ///< Value of the tensor in VERT direction
    // clang-format on

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
    Tensor3(const Tensor3<OtherT>& p)
        : c00(p.c00), c01(p.c01), c02(p.c02), c10(p.c10), c11(p.c11), c12(p.c12), c20(p.c20), c21(p.c21), c22(p.c22) {}

    /**
     * Construct isotropic tensor.
     * @param val value
     */
    Tensor3(const T& val) : c00(val), c01(0.), c02(0.), c10(0.), c11(val), c12(0.), c20(0.), c21(0.), c22(val) {}

    /**
     * Construct tensor with given diagonal values.
     * @param c00, c22 components
     */
    Tensor3(const T& c00, const T& c22) : c00(c00), c01(0.), c02(0.), c10(0.), c11(c00), c12(0.), c20(0.), c21(0.), c22(c22) {}

    /**
     * Construct tensors with given diagonal values.
     * @param c00, c11, c22 components
     */
    Tensor3(const T& c00, const T& c11, const T& c22)
        : c00(c00), c01(0.), c02(0.), c10(0.), c11(c11), c12(0.), c20(0.), c21(0.), c22(c22) {}

    /**
     * Construct tensors with given lateral Hermitian values.
     * @param c00, c11, c22, c01 components
     */
    Tensor3(const T& c00, const T& c11, const T& c22, const T& c01)
        : c00(c00), c01(c01), c02(0.), c10(conj(c01)), c11(c11), c12(0.), c20(0.), c21(0.), c22(c22) {}

    /**
     * Construct tensors with given lateral values.
     * @param c00, c11, c22, c01, c10 components
     */
    Tensor3(const T& c00, const T& c11, const T& c22, const T& c01, const T& c10)
        : c00(c00), c01(c01), c02(0.), c10(c10), c11(c11), c12(0.), c20(0.), c21(0.), c22(c22) {}

    /**
     * Construct tensors with given Hermitian values.
     * @param c00, c11, c22, c01 components
     */
    Tensor3(const T& c00, const T& c11, const T& c22, const T& c01, const T& c02, const T& c12)
        : c00(c00), c01(c01), c02(c02), c10(conj(c01)), c11(c11), c12(c12), c20(conj(c02)), c21(conj(c12)), c22(c22) {}

    /**
     * Construct tensors with given all values.
     * @param c00, c11, c22, c01, c02, c12 components
     */
    Tensor3(const T& c00,
            const T& c11,
            const T& c22,
            const T& c01,
            const T& c10,
            const T& c02,
            const T& c20,
            const T& c12,
            const T& c21)
        : c00(c00), c01(c01), c02(c02), c10(c10), c11(c11), c12(c12), c20(c20), c21(c21), c22(c22) {}

    /**
     * Construct tensor components given in std::pair.
     * @param comp components
     */
    template <typename T0, typename T1>
    Tensor3(const std::pair<T0, T1>& comp)
        : c00(comp.first), c01(0.), c02(0.), c10(0.), c11(comp.first), c12(0.), c20(0.), c21(0.), c22(comp.second) {}

    /**
     * Construct tensor from 2D tensor
     * \param tens tensor
     */
    Tensor3(const Tensor2<T>& tens)
        : c00(tens.c00), c01(0.), c02(0.), c10(0.), c11(tens.c00), c12(0.), c20(0.), c21(0.), c22(tens.c11) {}

    /**
     * Construct tensor from 2D vector
     * \param vec vector
     */
    Tensor3(const Vec<2, T>& vec) : c00(vec.c0), c01(0.), c02(0.), c10(0.), c11(vec.c0), c12(0.), c20(0.), c21(0.), c22(vec.c1) {}

    /**
     * Construct tensor from 3D vector
     * \param vec vector
     */
    Tensor3(const Vec<3, T>& vec) : c00(vec.c0), c01(0.), c02(0.), c10(0.), c11(vec.c1), c12(0.), c20(0.), c21(0.), c22(vec.c2) {}

    /**
     * Construct tensor from data
     * \param vec vector
     */
    Tensor3(const T* data)
        : c00(data[0]),
          c01(data[1]),
          c02(data[2]),
          c10(data[3]),
          c11(data[4]),
          c12(data[5]),
          c20(data[6]),
          c21(data[7]),
          c22(data[8]) {}

    /**
     * Get i-th component
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * \param i number of coordinate
     * \return i-th component
     */
    inline T& operator[](size_t i) {
        assert(i < 9);
        return *(&c00 + i);
    }

    /**
     * Get i-th component
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * \param i number of coordinate
     * \return i-th component
     */
    inline const T& operator[](size_t i) const {
        assert(i < 9);
        return *(&c00 + i);
    }

    /**
     * Get ij component
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * \param i,j coordinates
     * \return ij tensor component
     */
    inline T& operator()(size_t i, size_t j) {
        assert(i < 3 && j < 3);
        return *(&c00 + 3 * i + j);
    }

    /**
     * Get ij component
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * \param i,j coordinates
     * \return ij tensor component
     */
    inline const T& operator()(size_t i, size_t j) const {
        assert(i < 3 && j < 3);
        return *(&c00 + 3 * i + j);
    }

    /// Convert to std::tuple
    operator std::tuple<T, T, T, T, T, T, T, T, T>() const { return std::make_tuple(c00, c11, c22, c01, c10, c02, c20, c12, c21); }

    /**
     * Compare two tensors, this and @p p.
     * @param p tensor to compare
     * @return true only if this tensor and @p p have equals coordinates
     */
    template <typename OtherT> bool operator==(const Tensor3<OtherT>& p) const {
        return p.c00 == c00 && p.c01 == c01 && p.c02 == c02 && p.c10 == c10 && p.c11 == c11 && p.c12 == c12 && p.c20 == c20 &&
               p.c21 == c21 && p.c22 == c22;
    }

    /**
     * Check if two tensors, this and @p p are almost equal.
     * @param p tensors to compare
     * @return @c true only if this tensors and @p p have almost equals coordinates
     */
    template <typename OtherT> constexpr bool equals(const Tensor3<OtherT>& p) const {
        return is_zero(p.c00 - c00) && is_zero(p.c01 - c01) && is_zero(p.c02 - c02) && is_zero(p.c10 - c10) &&
               is_zero(p.c11 - c11) && is_zero(p.c12 - c12) && is_zero(p.c20 - c20) && is_zero(p.c21 - c21) && is_zero(p.c22 - c22);
    }

    /**
     * Compare two tensors, this and @p p.
     * @param p tensor to compare
     * @return true only if this tensor and @p p don't have equals coordinates
     */
    template <typename OtherT> bool operator!=(const Tensor3<OtherT>& p) const {
        return p.c00 != c00 || p.c01 != c01 || p.c02 != c02 || p.c10 != c10 || p.c11 != c11 || p.c12 != c12 || p.c20 != c20 ||
               p.c21 != c21 || p.c22 != c22;
    }

    /**
     * Calculate sum of two tensors, @c this and @p other.
     * @param other tensor to add, can have different data type (than result type will be found using C++ types promotions rules)
     * @return tensors sum
     */
    template <typename OtherT> auto operator+(const Tensor3<OtherT>& other) const -> Tensor3<decltype(c00 + other.c00)> {
        return Tensor3<decltype(this->c00 + other.c00)>(c00 + other.c00, c11 + other.c11, c22 + other.c22, c01 + other.c01,
                                                        c10 + other.c10, c02 + other.c02, c20 + other.c20, c12 + other.c12,
                                                        c21 + other.c21);
    }

    /**
     * Increase coordinates of this tensor by coordinates of other tensor @p other.
     * @param other tensor to add
     * @return *this (after increase)
     */
    Tensor3<T>& operator+=(const Tensor3<T>& other) {
        c00 += other.c00;
        c01 += other.c01;
        c02 += other.c02;
        c10 += other.c10;
        c11 += other.c11;
        c12 += other.c12;
        c20 += other.c20;
        c21 += other.c21;
        c22 += other.c22;
        return *this;
    }

    /**
     * Calculate difference of two tensors, @c this and @p other.
     * @param other tensor to subtract from this, can have different data type (than result type will be found using C++ types
     * promotions rules)
     * @return tensors difference
     */
    template <typename OtherT> auto operator-(const Tensor3<OtherT>& other) const -> Tensor3<decltype(c00 - other.c00)> {
        return Tensor3<decltype(this->c00 - other.c00)>(c00 - other.c00, c11 - other.c11, c22 - other.c22, c01 - other.c01,
                                                        c10 - other.c10, c02 - other.c02, c20 - other.c20, c12 - other.c12,
                                                        c21 - other.c21);
    }

    /**
     * Decrease coordinates of this tensor by coordinates of other tensor @p other.
     * @param other tensor to subtract
     * @return *this (after decrease)
     */
    Tensor3<T>& operator-=(const Tensor3<T>& other) {
        c00 -= other.c00;
        c01 -= other.c01;
        c02 -= other.c02;
        c10 -= other.c10;
        c11 -= other.c11;
        c12 -= other.c12;
        c20 -= other.c20;
        c21 -= other.c21;
        c22 -= other.c22;
        return *this;
    }

    /**
     * Calculate this tensor multiplied by scalar @p scale.
     * @param scale scalar
     * @return this tensor multiplied by scalar
     */
    template <typename OtherT> auto operator*(const OtherT scale) const -> Tensor3<decltype(c00 * scale)> {
        return Tensor3<decltype(c00 * scale)>(c00 * scale, c11 * scale, c22 * scale, c01 * scale, c10 * scale, c02 * scale,
                                              c20 * scale, c12 * scale, c21 * scale);
    }

    /**
     * Multiple coordinates of this tensor by @p scalar.
     * @param scalar scalar
     * @return *this (after scale)
     */
    Tensor3<T>& operator*=(const T scalar) {
        c00 *= scalar;
        c01 *= scalar;
        c02 *= scalar;
        c10 *= scalar;
        c11 *= scalar;
        c12 *= scalar;
        c20 *= scalar;
        c21 *= scalar;
        c22 *= scalar;
        return *this;
    }

    /**
     * Calculate this tensor divided by scalar @p scale.
     * @param scale scalar
     * @return this tensor divided by scalar
     */
    Tensor3<T> operator/(const T scale) const {
        return Tensor3<decltype(c00 / scale)>(c00 / scale, c11 / scale, c22 / scale, c01 / scale, c10 / scale, c02 / scale,
                                              c20 / scale, c12 / scale, c21 / scale);
    }

    /**
     * Divide coordinates of this tensor by @p scalar.
     * @param scalar scalar
     * @return *this (after divide)
     */
    Tensor3<T>& operator/=(const T scalar) {
        c00 /= scalar;
        c01 /= scalar;
        c02 /= scalar;
        c10 /= scalar;
        c11 /= scalar;
        c12 /= scalar;
        c20 /= scalar;
        c21 /= scalar;
        c22 /= scalar;
        return *this;
    }

    /**
     * Calculate tensor opposite to this.
     * @return Tensor3<T>(-c00, -c11)
     */
    Tensor3<T> operator-() const { return Tensor3<T>(-c00, -c11, -c22, -c01, -c10, -c02, -c20, -c12, -c21); }

    /**
     * Square each component of tensor
     * \return squared tensor
     */
    Tensor3<T> sqr() const {
        return Tensor3<T>(c00 * c00 + c01 * c10 + c02 * c20,  // c00
                          c10 * c01 + c11 * c11 + c12 * c21,  // c11
                          c20 * c02 + c21 * c12 + c22 * c22,  // c22
                          c00 * c01 + c01 * c11 + c02 * c21,  // c01
                          c10 * c00 + c11 * c10 + c12 * c20,  // c10
                          c00 * c02 + c01 * c12 + c02 * c22,  // c02
                          c20 * c00 + c21 * c10 + c22 * c20,  // c20
                          c10 * c02 + c11 * c12 + c12 * c22,  // c12
                          c20 * c01 + c21 * c11 + c22 * c21   // c21
        );
    }

    /**
     * Power of  tensor
     * \return squared tensor
     */
    Tensor3<T> pow(int n) const {
        if (n < 0) {
            return inv().pow(-n);
        } else if (n == 0) {
            return Tensor3<T>(1.);
        } else if (n == 1) {
            return *this;
        } else if (n == 2) {
            return sqr();
        } else if (n % 2 == 0) {
            return sqr().pow(n / 2);
        } else {
            Tensor3<T> a = sqr().pow(n / 2), b(*this);
            return Tensor3<T>(a.c00 * b.c00 + a.c01 * b.c10 + a.c02 * b.c20,  // c00
                              a.c10 * b.c01 + a.c11 * b.c11 + a.c12 * b.c21,  // c11
                              a.c20 * b.c02 + a.c21 * b.c12 + a.c22 * b.c22,  // c22
                              a.c00 * b.c01 + a.c01 * b.c11 + a.c02 * b.c21,  // c01
                              a.c10 * b.c00 + a.c11 * b.c10 + a.c12 * b.c20,  // c10
                              a.c00 * b.c02 + a.c01 * b.c12 + a.c02 * b.c22,  // c02
                              a.c20 * b.c00 + a.c21 * b.c10 + a.c22 * b.c20,  // c20
                              a.c10 * b.c02 + a.c11 * b.c12 + a.c12 * b.c22,  // c12
                              a.c20 * b.c01 + a.c21 * b.c11 + a.c22 * b.c21   // c21
            );
        }
    }

    // /**
    //  * Square root of each component of tensor
    //  * \return squared tensor
    //  */
    // Tensor3<T> sqrt() const {
    //     TODO
    // }

    /**
     * Inverse of the tensor
     * https://www.wikihow.com/Find-the-Inverse-of-a-3x3-Matrix
     * \return inverse of this
     */
    Tensor3<T> inv() const {
        // clang-format off
        T a00 = c11*c22 - c12*c21,  a01 = c02*c21 - c01*c22,  a02 = c01*c12 - c02*c11,
          a10 = c12*c20 - c10*c22,  a11 = c00*c22 - c02*c20,  a12 = c02*c10 - c00*c12,
          a20 = c10*c21 - c11*c20,  a21 = c01*c20 - c00*c21,  a22 = c00*c11 - c01*c10;
        // clang-format on

        T det = c00 * c11 * c22 + c01 * c12 * c20 + c02 * c10 * c21 - c02 * c11 * c20 - c00 * c12 * c21 - c10 * c01 * c22;
        return Tensor3<T>(a00 / det, a11 / det, a22 / det, a01 / det, a10 / det, a02 / det, a20 / det, a12 / det, a21 / det);
    }

    /**
     * Print tensor to stream using format (where c00 and c11 are tensor components): [c00, c11]
     * @param out print destination, output stream
     * @param to_print tensor to print
     * @return out stream
     */
    friend inline std::ostream& operator<<(std::ostream& out, const Tensor3<T>& to_print) {
        return out << '[' << str(to_print.c00) << ", " << str(to_print.c01) << ", " << str(to_print.c02) << "; "
                   << str(to_print.c10) << ", " << str(to_print.c11) << ", " << str(to_print.c12) << "; " << str(to_print.c20)
                   << ", " << str(to_print.c21) << ", " << str(to_print.c22) << "]";
    }
};

/**
 * Calculate this tensor multiplied by scalar \p scale.
 * \param scale scalar
 * \param tensor tensor
 * \return this tensor multiplied by scalar
 */
template <typename T, typename OtherT> auto operator*(const OtherT scale, const Tensor3<T>& tensor) -> decltype(tensor * scale) {
    return tensor * scale;
}

/**
 * Calculate tensor conjugate.
 * @param v a tensor
 * @return conjugate tensor
 */
template <typename T> inline Tensor3<T> conj(const Tensor3<T>& v) {
    return Tensor3<T>(conj(v.c00), conj(v.c11), conj(v.c22), conj(v.c01), conj(v.c10), conj(v.c02), conj(v.c20), conj(v.c12),
                      conj(v.c21));
}

/// Specialization of NaNImpl which add support for 3D tensors.
template <typename T> struct NaNImpl<Tensor3<T>> {
    static constexpr Tensor3<T> get() { return Tensor3<T>(NaN<T>()); }
};

/// Specialization of ZeroImpl which add support for 3D vectors.
template <typename T> struct ZeroImpl<Tensor3<T>> {
    static constexpr Tensor3<T> get() { return Tensor3<T>(0.); }
};

/// Check if the tensor is almost zero
/// \param v tensor to verify
template <typename T> inline bool is_zero(const Tensor3<T>& v) {
    return is_zero(v.c00) && is_zero(v.c01) && is_zero(v.c02) && is_zero(v.c10) && is_zero(v.c11) && is_zero(v.c12) &&
           is_zero(v.c20) && is_zero(v.c21) && is_zero(v.c22);
}

/*
PLASK_API_EXTERN_TEMPLATE_STRUCT(Tensor3<double>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(Tensor3< std::complex<double> >)
*/

template <typename T>
inline bool isnan(plask::Tensor3<T> tens) {
    return isnan(tens.c00) || isnan(tens.c01) || isnan(tens.c02) || isnan(tens.c10) || isnan(tens.c11) || isnan(tens.c12) ||
           isnan(tens.c20) || isnan(tens.c21) || isnan(tens.c22);
}

}  // namespace plask

namespace std {

template <typename T> plask::Tensor3<T> pow(plask::Tensor3<T> tens, int n) { return tens.pow(n); }

}  // namespace std

#endif  // PLASK__TENSOR3_H
