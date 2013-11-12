#ifndef PLASK__VECTOR2D_H
#define PLASK__VECTOR2D_H

/** @file
This file contains implementation of vector in 2D space.
*/

#include <iostream>

#include "../math.h"
#include <plask/exceptions.h>

#include "common.h"
#include <cassert>

namespace plask {

/**
 * Vector in 2D space.
 */
template <typename T>
struct Vec<2,T> {

    static const int DIMS = 2;

    T c0, c1;

    T& tran() { return c0; }
    constexpr const T& tran() const { return c0; }

    T& vert() { return c1; }
    constexpr const T& vert() const { return c1; }

    // radial coordinates
    T& rad_r() { return c0; }
    constexpr const T& rad_r() const { return c0; }
    T& rad_z() { return c1; }
    constexpr const T& rad_z() const { return c1; }

    // for surface-emitting lasers (z-axis up)
    T& se_y() { return c0; }
    constexpr const T& se_y() const { return c0; }
    T& se_z() { return c1; }
    constexpr const T& se_z() const { return c1; }

    // for surface-emitting lasers (z-axis up)
    T& zup_y() { return c0; }
    constexpr const T& z_up_y() const { return c0; }
    T& zup_z() { return c1; }
    constexpr const T& z_up_z() const { return c1; }

    // for edge emitting lasers (y-axis up), we keep the coordinates right-handed
    T& ee_x() { return c0; }
    constexpr const T& ee_x() const { return c0; }
    T& ee_y() { return c1; }
    constexpr const T& ee_y() const { return c1; }

    // for edge emitting lasers (y-axis up), we keep the coordinates right-handed
    T& yup_x() { return c0; }
    constexpr const T& y_up_x() const { return c0; }
    T& yup_y() { return c1; }
    constexpr const T& y_up_y() const { return c1; }


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
    constexpr Vec(const Vec<2,OtherT>& p): c0(p.c0), c1(p.c1) {}

    /**
     * Construct vector with given components.
     * @param c0__tran, c1__up components
     */
    constexpr Vec(const T& c0__tran, const T& c1__up): c0(c0__tran), c1(c1__up) {}

    /**
     * Construct vector components given in std::pair.
     * @param comp components
     */
    template <typename T0, typename T1>
    constexpr Vec(const std::pair<T0,T1>& comp): c0(comp.first), c1(comp.second) {}

    /**
     * Construct vector with components read from input iterator (including C array).
     * @param inputIt input iterator with minimum 3 objects available
     * @tparam InputIteratorType input iterator type, must allow for postincrementation and derefrence operation
     */
    template <typename InputIteratorType>
    static inline Vec<2,T> fromIterator(InputIteratorType inputIt) {
        Vec<2,T> result;
        result.c0 = *inputIt;
        result.c1 = *++inputIt;
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
    iterator end() { return &c0 + 2; }

    /**
     * Get end const iterator over components.
     * @return end const iterator over components
     */
    const_iterator end() const { return &c0 + 2; }

    /**
     * Compare two vectors, this and @p p.
     * @param p vector to compare
     * @return true only if this vector and @p p have equals coordinates
     */
    template <typename OtherT>
    constexpr bool operator==(const Vec<2,OtherT>& p) const { return p.c0 == c0 && p.c1 == c1; }

    /**
     * Compare two vectors, this and @p p.
     * @param p vector to compare
     * @param abs_supremum maximal allowed difference for one coordinate
     * @return @c true only if this vector and @p p have almost equals coordinates
     */
    constexpr bool equal(const Vec<2, T>& p, const T& abs_supremum = SMALL) const { return is_zero(p.c0 - c0, abs_supremum) && is_zero(p.c1 - c1, abs_supremum); }

    /**
     * Compare two vectors, this and @p p.
     * @param p vector to compare
     * @return true only if this vector and @p p don't have equals coordinates
     */
    template <typename OtherT>
    constexpr bool operator!=(const Vec<2,OtherT>& p) const { return p.c0 != c0 || p.c1 != c1; }

    /**
     * Get i-th component
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * @param i number of coordinate
     * @return i-th component
     */
    inline T& operator[](size_t i) {
        assert(i < 2);
        return *(&c0 + i);
    }

    /**
     * Get i-th component
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * @param i number of coordinate
     * @return i-th component
     */
    inline const T& operator[](size_t i) const {
        assert(i < 2);
        return *(&c0 + i);
    }

    /**
     * Calculate sum of two vectors, @c this and @p other.
     * @param other vector to add, can have different data type (than result type will be found using C++ types promotions rules)
     * @return vectors sum
     */
    template <typename OtherT>
    constexpr auto operator+(const Vec<2,OtherT>& other) const -> Vec<2,decltype(c0 + other.c0)> {
        return Vec<2,decltype(this->c0 + other.c0)>(c0 + other.c0, c1 + other.c1);
    }

    /**
     * Increase coordinates of this vector by coordinates of other vector @p other.
     * @param other vector to add
     * @return *this (after increase)
     */
    Vec<2,T>& operator+=(const Vec<2,T>& other) {
        c0 += other.c0;
        c1 += other.c1;
        return *this;
    }

    /**
     * Calculate difference of two vectors, @c this and @p other.
     * @param other vector to subtract from this, can have different data type (than result type will be found using C++ types promotions rules)
     * @return vectors difference
     */
    template <typename OtherT>
    constexpr auto operator-(const Vec<2,OtherT>& other) const -> Vec<2,decltype(c0 - other.c0)> {
        return Vec<2,decltype(this->c0 - other.c0)>(c0 - other.c0, c1 - other.c1);
    }

    /**
     * Decrease coordinates of this vector by coordinates of other vector @p other.
     * @param other vector to subtract
     * @return *this (after decrease)
     */
    Vec<2,T>& operator-=(const Vec<2,T>& other) {
        c0 -= other.c0;
        c1 -= other.c1;
        return *this;
    }

    /**
     * Calculate this vector multiplied by scalar @p scale.
     * @param scale scalar
     * @return this vector multiplied by scalar
     */
    template <typename OtherT>
    constexpr auto operator*(const OtherT scale) const -> Vec<2,decltype(c0*scale)> {
        return Vec<2,decltype(c0*scale)>(c0 * scale, c1 * scale);
    }

    /**
     * Multiple coordinates of this vector by @p scalar.
     * @param scalar scalar
     * @return *this (after scale)
     */
    Vec<2,T>& operator*=(const T scalar) {
        c0 *= scalar;
        c1 *= scalar;
        return *this;
    }

    /**
     * Calculate this vector divided by scalar @p scale.
     * @param scale scalar
     * @return this vector divided by scalar
     */
    constexpr Vec<2,T> operator/(const T scale) const { return Vec<2,T>(c0 / scale, c1 / scale); }

    /**
     * Divide coordinates of this vector by @p scalar.
     * @param scalar scalar
     * @return *this (after divide)
     */
    Vec<2,T>& operator/=(const T scalar) {
        c0 /= scalar;
        c1 /= scalar;
        return *this;
    }

    /**
     * Calculate vector opposite to this.
     * @return Vec<2,T>(-c0, -c1)
     */
    constexpr Vec<2,T> operator-() const {
        return Vec<2,T>(-c0, -c1);
    }

    /**
     * Change i-th coordinate to oposite.
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * @param i number of coordinate
     */
    inline void flip(size_t i) {
        assert(i < 2);
        operator[](i) = -operator[](i);
    }

    /**
     * Get vector similar to this but with changed i-th component to oposite.
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * @param i number of coordinate
     * @return vector similar to this but with changed i-th component to oposite
     */
    inline Vec<2,T> fliped(size_t i) {
        Vec<2,T> res = *this;
        res.flip(i);
        return res;
    }

    /**
     * Print vector to stream using format (where c0 and c1 are vector components): [c0, c1]
     * @param out print destination, output stream
     * @param to_print vector to print
     * @return out stream
     */
    friend inline std::ostream& operator<<(std::ostream& out, const Vec<2,T>& to_print) {
        return out << '[' << to_print.c0 << ", " << to_print.c1 << ']';
    }

    /**
     * A lexical comparison of two vectors, allow to use vector in std::set and std::map as key type.
     * @param v vectors to compare
     * @return @c true only if @c this is smaller than the @p v
     */
    template<class OT> inline
    bool operator< (Vec<2, OT> const& v) const {
        if (this->c0 < v.c0) return true;
        if (this->c0 > v.c0) return false;
        return this->c1 < v.c1;
    }
};

/**
 * Calculate vector conjugate.
 * @param v a vector
 * @return conjugate vector
 */
template <typename T>
inline constexpr Vec<2,T> conj(const Vec<2,T>& v) { return Vec<2,T>(conj(v.c0), conj(v.c1)); }

/**
 * Compute dot product of two vectors @p v1 and @p v2.
 * @param v1 first vector
 * @param v2 second vector
 * @return dot product v1·v2
 */
template <typename T1, typename T2>
inline auto dot(const Vec<2,T1>& v1, const Vec<2,T2>& v2) -> decltype(v1.c0*v2.c0) {
    return fma(v1.c0, v2.c0, v1.c1 * v2.c1);
}

/**
 * Compute dot product of two vectors @p v1 and @p v2.
 * @param v1 first vector
 * @param v2 second vector
 * @return dot product v1·v2
 */
template <>
inline auto dot(const Vec<2,double>& v1, const Vec<2,complex<double>>& v2) -> decltype(v1.c0*v2.c0) {
    return fma(v1.c0, conj(v2.c0), v1.c1 * conj(v2.c1));
}

/**
 * Compute dot product of two vectors @p v1 and @p v2.
 * @param v1 first vector
 * @param v2 second vector
 * @return dot product v1·v2
 */
template <>
inline auto dot(const Vec<2,complex<double>>& v1, const Vec<2,complex<double>>& v2) -> decltype(v1.c0*v2.c0) {
    return fma(v1.c0, conj(v2.c0), v1.c1 * conj(v2.c1));
}

/**
 * Helper to create 2D vector.
 * @param c0__tran, c1__up vector coordinates
 * @return constructed vector
 */
template <typename T>
inline constexpr Vec<2,T> vec(const T c0__tran, const T c1__up) {
    return Vec<2,T>(c0__tran, c1__up);
}

} //namespace plask

#endif // PLASK__VECTOR2D_H
