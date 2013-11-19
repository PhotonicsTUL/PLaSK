#ifndef PLASK__VECTORCART3D_H
#define PLASK__VECTORCART3D_H

/** @file
This file contains implementation of vector in 3D space.
*/

#include <iostream>

#include "../math.h"
#include <plask/exceptions.h>

#include "common.h"

namespace plask {

/**
 * Vector in 3D space.
 */
template <typename T>
struct Vec<3,T> {

    static const int DIMS = 3;

    /// Vector components
    T c0, c1, c2;

    T& lon() { return c0; }
    constexpr const T& lon() const { return c0; }

    T& tran() { return c1; }
    constexpr const T& tran() const { return c1; }

    T& vert() { return c2; }
    constexpr const T& vert() const { return c2; }

    // radial coordinates
    T& rad_p() { return c0; }
    constexpr const T& rad_p() const { return c0; }

    T& rad_r() { return c1; }
    constexpr const T& rad_r() const { return c1; }

    T& rad_z() { return c2; }
    constexpr const T& rad_z() const { return c2; }

    // for surface-emitting lasers (z-axis up)
    T& se_x() { return c0; }
    constexpr const T& se_x() const { return c0; }

    T& se_y() { return c1; }
    constexpr const T& se_y() const { return c1; }

    T& se_z() { return c2; }
    constexpr const T& se_z() const { return c2; }

    // for surface-emitting lasers (z-axis up)
    T& zup_x() { return c0; }
    constexpr const T& z_up_x() const { return c0; }

    T& zup_y() { return c1; }
    constexpr const T& z_up_y() const { return c1; }

    T& zup_z() { return c2; }
    constexpr const T& z_up_z() const { return c2; }

    // for edge emitting lasers (y-axis up), we keep the coordinates right-handed
    T& ee_z() { return c0; }
    constexpr const T& ee_z() const { return c0; }

    T& ee_x() { return c1; }
    constexpr const T& ee_x() const { return c1; }

    T& ee_y() { return c2; }
    constexpr const T& ee_y() const { return c2; }

    // for edge emitting lasers (y-axis up), we keep the coordinates right-handed
    T& yup_z() { return c0; }
    constexpr const T& y_up_z() const { return c0; }

    T& yup_x() { return c1; }
    constexpr const T& y_up_x() const { return c1; }

    T& yup_y() { return c2; }
    constexpr const T& y_up_y() const { return c2; }

    /**
     * Type of iterator over components.
     */
    typedef T* iterator;

    /**
     * Type of const iterator over components.
     */
    typedef const T* const_iterator;

    /// Construct uninitialized vector.
    Vec() {}

    /**
     * Copy constructor from all other 2D vectors.
     * @param p vector to copy from
     */
    template <typename OtherT>
    constexpr Vec(const Vec<3,OtherT>& p): c0(p.c0), c1(p.c1), c2(p.c2) {}

    /**
     * Construct vector with given components.
     * @param c0__lon, c1__tran, c2__up components
     */
    constexpr Vec(const T& c0__lon, const T& c1__tran, const T& c2__up): c0(c0__lon), c1(c1__tran), c2(c2__up) {}

    /**
     * Construct vector components given in std::tuple.
     * @param comp components
     */
    template <typename T0, typename T1, typename T2>
    constexpr Vec(const std::tuple<T0,T1,T2>& comp): c0(std::get<0>(comp)), c1(std::get<1>(comp)), c2(std::get<2>(comp)) {}

    /**
     * Construct vector with components read from input iterator (including C array).
     * @param inputIt input iterator with minimum 3 objects available
     * @tparam InputIteratorType input iterator type, must allow for postincrementation and derefrence operation
     */
    template <typename InputIteratorType>
    static inline Vec<3,T> fromIterator(InputIteratorType inputIt) {
        Vec<3,T> result;
        result.c0 = *inputIt;
        result.c1 = *++inputIt;
        result.c2 = *++inputIt;
        return result;
    }

    /**
     * Get begin iterator over components.
     * @return begin iterator over components
     */
    iterator begin() { return &c0; }

    /**
     * Get begin const iterator over components.
     * @return begin const iterator over components
     */
    const_iterator begin() const { return &c0; }

    /**
     * Get end iterator over components.
     * @return end iterator over components
     */
    iterator end() { return &c0 + 3; }

    /**
     * Get end const iterator over components.
     * @return end const iterator over components
     */
    const_iterator end() const { return &c0 + 3; }

    /**
     * Compare two vectors, @c this and @p p.
     * @param p vector to compare
     * @return true only if this vector and @p p have equals coordinates
     */
    template <typename OtherT>
    constexpr bool operator==(const Vec<3,OtherT>& p) const { return p.c0 == c0 && p.c1 == c1 && p.c2 == c2; }

    /**
     * Compare two vectors, this and @p p.
     * @param p vector to compare
     * @param abs_supremum maximal allowed difference for one coordinate
     * @return @c true only if this vector and @p p have almost equals coordinates
     */
    constexpr bool equal(const Vec<3, T>& p, const T& abs_supremum = SMALL) const { return is_zero(p.c0 - c0, abs_supremum) && is_zero(p.c1 - c1, abs_supremum) && is_zero(p.c2 - c2, abs_supremum); }

    /**
     * Compare two vectors, @c this and @p p.
     * @param p vector to compare
     * @return true only if this vector and @p p don't have equals coordinates
     */
    template <typename OtherT>
    constexpr bool operator!=(const Vec<3,OtherT>& p) const { return p.c0 != c0 || p.c1 != c1 || p.c2 != c2; }

    /**
     * Get i-th component
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * @param i number of coordinate
     * @return i-th component
     */
    inline T& operator[](size_t i) {
        assert(i < 3);
        return *(&c0 + i);
    }

    /**
     * Get i-th component
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * @param i number of coordinate
     * @return i-th component
     */
    inline const T& operator[](size_t i) const {
        assert(i < 3);
        return *(&c0 + i);
    }

    /**
     * Calculate sum of two vectors, @c this and @p other.
     * @param other vector to add, can have different data type (than result type will be found using C++ types promotions rules)
     * @return vectors sum
     */
    template <typename OtherT>
    constexpr auto operator+(const Vec<3,OtherT>& other) const -> Vec<3,decltype(c0 + other.c0)> {
        return Vec<3,decltype(this->c0 + other.c0)>(c0 + other.c0, c1 + other.c1, c2 + other.c2);
    }

    /**
     * Increase coordinates of this vector by coordinates of other vector @p other.
     * @param other vector to add
     * @return *this (after increase)
     */
    Vec<3,T>& operator+=(const Vec<3,T>& other) {
        c0 += other.c0;
        c1 += other.c1;
        c2 += other.c2;
        return *this;
    }

    /**
     * Calculate difference of two vectors, @c this and @p other.
     * @param other vector to subtract from this, can have different data type (than result type will be found using C++ types promotions rules)
     * @return vectors difference
     */
    template <typename OtherT>
    constexpr auto operator-(const Vec<3,OtherT>& other) const -> Vec<3,decltype(c0 - other.c0)> {
        return Vec<3, decltype(this->c0 - other.c0)>(c0 - other. c0, c1 - other. c1, c2 - other.c2);
    }

    /**
     * Decrease coordinates of this vector by coordinates of other vector @p other.
     * @param other vector to subtract
     * @return *this (after decrease)
     */
    Vec<3,T>& operator-=(const Vec<3,T>& other) {
        c0 -= other.c0;
        c1 -= other.c1;
        c2 -= other.c2;
        return *this;
    }

    /**
     * Calculate this vector multiplied by scalar @p scale.
     * @param scale scalar
     * @return this vector multiplied by scalar
     */
    template <typename OtherT>
    constexpr auto operator*(const OtherT scale) const -> Vec<3,decltype(c0*scale)> {
        return Vec<3,decltype(c0*scale)>(c0 * scale, c1 * scale, c2 * scale);
    }

    /**
     * Multiple coordinates of this vector by @p scalar.
     * @param scalar scalar
     * @return *this (after scale)
     */
    Vec<3,T>& operator*=(const T scalar) {
        c0 *= scalar;
        c1 *= scalar;
        c2 *= scalar;
        return *this;
    }

    /**
     * Calculate this vector divided by @p scalar.
     * @param scalar scalar
     * @return this vector divided by @p scalar
     */
    constexpr Vec<3,T> operator/(const T scalar) const { return Vec<3,T>(c0 / scalar, c1 / scalar, c2 / scalar); }

    /**
     * Divide coordinates of this vector by @p scalar.
     * @param scalar scalar
     * @return *this (after divide)
     */
    Vec<3,T>& operator/=(const T scalar) {
        c0 /= scalar;
        c1 /= scalar;
        c2 /= scalar;
        return *this;
    }

    /**
     * Calculate vector opposite to this.
     * @return Vec<3,T>(-c0, -c1, -c2)
     */
    constexpr Vec<3,T> operator-() const {
        return Vec<3,T>(-c0, -c1, -c2);
    }

    /**
     * Change i-th coordinate to oposite.
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * @param i number of coordinate
     */
    inline void flip(size_t i) {
        assert(i < 3);
        operator[](i) = -operator[](i);
    }

    /**
     * Get vector similar to this but with changed i-th component to oposite.
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * @param i number of coordinate
     * @return vector similar to this but with changed i-th component to oposite
     */
    inline Vec<3,T> fliped(size_t i) {
        Vec<3,T> res = *this;
        res.flip(i);
        return res;
    }

    /**
     * Print vector to stream using format (where c0, c1 and c2 are vector coordinates): [c0, c1, c2]
     * @param out print destination, output stream
     * @param to_print vector to print
     * @return out stream
     */
    friend inline std::ostream& operator<<(std::ostream& out, const Vec<3,T>& to_print) {
        return out << '[' << to_print.c0 << ", " << to_print.c1 << ", " << to_print.c2 << ']';
    }

    /**
     * A lexical comparison of two vectors, allow to use vector in std::set and std::map as key type.
     *
     * It supports NaN-s (which, due to this method, is greater than all other numbers).
     * @param v vectors to compare
     * @return @c true only if @c this is smaller than the @p v
     */
    template<class OT>
    bool operator< (Vec<3, OT> const& v) const {
        if (dbl_compare_lt(this->c0, v.c0)) return true;
        if (dbl_compare_gt(this->c0, v.c0)) return false;
        if (dbl_compare_lt(this->c1, v.c1)) return true;
        if (dbl_compare_gt(this->c1, v.c1)) return false;
        return dbl_compare_lt(this->c2, v.c2);
    }


};

/**
 * Calculate vector conjugate.
 * @param v a vector
 * @return conjugate vector
 */
template <typename T>
inline constexpr Vec<3,T> conj(const Vec<3,T>& v) { return Vec<3,T>(conj(v.c0), conj(v.c1), conj(v.c2)); }

/**
 * Compute dot product of two vectors @p v1 and @p v2.
 * @param v1 first vector
 * @param v2 second vector
 * @return dot product v1·v2
 */
template <typename T1, typename T2>
inline auto dot(const Vec<3,T1>& v1, const Vec<3,T2>& v2) -> decltype(v1.c0*v2.c0) {
    return fma(v1.c0, v2.c0, fma(v1.c1, v2.c1, v1.c2 * v2.c2));
}

/**
 * Compute dot product of two vectors @p v1 and @p v2.
 * @param v1 first vector
 * @param v2 second vector
 * @return dot product v1·v2
 */
template <>
inline auto dot(const Vec<3,double>& v1, const Vec<3,complex<double>>& v2) -> decltype(v1.c0*v2.c0) {
    return fma(v1.c0, conj(v2.c0), fma(v1.c1, conj(v2.c1), v1.c2 * conj(v2.c2)));
}

/**
 * Compute dot product of two vectors @p v1 and @p v2.
 * @param v1 first vector
 * @param v2 second vector
 * @return dot product v1·v2
 */
template <>
inline auto dot(const Vec<3,complex<double>>& v1, const Vec<3,complex<double>>& v2) -> decltype(v1.c0*v2.c0) {
    //v1.c0 * conj(v2.c0) + v1.c1 * conj(v2.c1) + v1.c2 * conj(v2.c2);
    return fma(v1.c0, conj(v2.c0), fma(v1.c1, conj(v2.c1), v1.c2 * conj(v2.c2)));
}

/**
 * Helper to create 3D vector.
 * @param c0__lon, c1__tran, c2__up vector coordinates
 * @return constructed vector
 */
template <typename T>
inline constexpr Vec<3,T> vec(const T c0__lon, const T c1__tran, const T c2__up) {
    return Vec<3,T>(c0__lon, c1__tran, c2__up);
}

} //namespace plask

#endif // PLASK__VECTORCART3D_H
