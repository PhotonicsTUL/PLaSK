/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2024 Lodz University of Technology
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
#ifndef PLASK_VECTOR_LATERAL_HPP
#define PLASK_VECTOR_LATERAL_HPP

#include "../math.hpp"
#include "../vec.hpp"

namespace plask {

template <typename T>
struct LateralVec {

    static const int DIMS = 2;

    T c0, c1;

    T& lon() { return c0; }
    constexpr const T& tran() const { return c0; }

    T& tran() { return c1; }
    constexpr const T& vert() const { return c1; }

    /**
     * Type of iterator over components.
     */
    typedef T* iterator;

    /**
     * Type of const iterator over components.
     */
    typedef const T* const_iterator;

    /// Construct uninitialized vector.
    LateralVec() {}

    /**
     * Copy constructor from all other lateral vectors.
     * \param p vector to copy from
     */
    template <typename OtherT>
    constexpr LateralVec(const LateralVec<OtherT>& p): c0(p.c0), c1(p.c1) {}

    /**
     * Copy constructor from 2D vectors.
     * \param p vector to copy from
     */
PLASK_NO_CONVERSION_WARNING_BEGIN
    template <int other_dim, typename OtherT>
    constexpr LateralVec(const Vec<other_dim,OtherT>& p): c0(p.c0), c1(p.c1) {}
PLASK_NO_WARNING_END

    /**
     * Construct vector with given components.
     * \param c0, c1 components
     */
    constexpr LateralVec(T c0, T c1): c0(c0), c1(c1) {}

    /**
     * Construct vector components given in std::pair.
     * \param comp components
     */
    template <typename T0, typename T1>
    constexpr LateralVec(const std::pair<T0,T1>& comp): c0(comp.first), c1(comp.second) {}

    /**
     * Construct vector with components read from input iterator (including C array).
     * \param inputIt input iterator with minimum 3 objects available
     * \tparam InputIteratorType input iterator type, must allow for postincrementation and dereference operation
     */
    template <typename InputIteratorType>
    static inline LateralVec<T> fromIterator(InputIteratorType inputIt) {
        LateralVec<T> result;
        result.c0 = T(*inputIt);
        result.c1 = T(*++inputIt);
        return result;
    }

    /**
     * Get begin iterator over components.
     * \return begin iterator over components
     */
    iterator begin() { return &c0; }

    /**
     * Get begin const iterator over components.
     * \return begin const iterator over components
     */
    const_iterator begin() const { return &c0; }

    /**
     * Get end iterator over components.
     * \return end iterator over components
     */
    iterator end() { return &c0 + 2; }

    /**
     * Get end const iterator over components.
     * \return end const iterator over components
     */
    const_iterator end() const { return &c0 + 2; }

    /**
     * Compare two vectors, this and \p p.
     * \param p vector to compare
     * \return true only if this vector and \p p have equals coordinates
     */
    template <typename OtherT>
    constexpr bool operator==(const LateralVec<OtherT>& p) const { return p.c0 == c0 && p.c1 == c1; }

    /**
     * Check if two vectors, this and \p p are almost equal.
     * \param p vector to compare
     * \param abs_supremum maximal allowed difference for one coordinate
     * \return \c true only if this vector and \p p have almost equals coordinates
     */
    template <typename OtherT, typename SuprType>
    constexpr bool equals(const LateralVec< OtherT>& p, const SuprType& abs_supremum) const {
        return is_zero(p.c0 - c0, abs_supremum) && is_zero(p.c1 - c1, abs_supremum); }

    /**
     * Check if two vectors, this and \p p are almost equal.
     * \param p vector to compare
     * \return \c true only if this vector and \p p have almost equals coordinates
     */
    template <typename OtherT>
    constexpr bool equals(const LateralVec< OtherT>& p) const {
        return is_zero(p.c0 - c0) && is_zero(p.c1 - c1); }

    /**
     * Compare two vectors, this and \p p.
     * \param p vector to compare
     * \return true only if this vector and \p p don't have equals coordinates
     */
    template <typename OtherT>
    constexpr bool operator!=(const LateralVec<OtherT>& p) const { return p.c0 != c0 || p.c1 != c1; }

    /**
     * Get i-th component
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * \param i number of coordinate
     * \return i-th component
     */
    inline T& operator[](size_t i) {
        assert(i < 2);
        return *(&c0 + i);
    }

    /**
     * Get i-th component
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * \param i number of coordinate
     * \return i-th component
     */
    inline const T& operator[](size_t i) const {
        assert(i < 2);
        return *(&c0 + i);
    }

    /**
     * Calculate sum of two vectors, \c this and \p other.
     * \param other vector to add, can have different data type (than result type will be found using C++ types promotions rules)
     * \return vectors sum
     */
    template <typename OtherT>
    constexpr auto operator+(const LateralVec<OtherT>& other) const -> LateralVec<decltype(c0 + other.c0)> {
        return LateralVec<decltype(this->c0 + other.c0)>(c0 + other.c0, c1 + other.c1);
    }

    /**
     * Increase coordinates of this vector by coordinates of other vector \p other.
     * \param other vector to add
     * \return *this (after increase)
     */
    LateralVec<T>& operator+=(const LateralVec<T>& other) {
        c0 += other.c0;
        c1 += other.c1;
        return *this;
    }

    /**
     * Calculate difference of two vectors, \c this and \p other.
     * \param other vector to subtract from this, can have different data type (than result type will be found using C++ types promotions rules)
     * \return vectors difference
     */
    template <typename OtherT>
    constexpr auto operator-(const LateralVec<OtherT>& other) const -> LateralVec<decltype(c0 - other.c0)> {
        return LateralVec<decltype(this->c0 - other.c0)>(c0 - other.c0, c1 - other.c1);
    }

    /**
     * Decrease coordinates of this vector by coordinates of other vector \p other.
     * \param other vector to subtract
     * \return *this (after decrease)
     */
    LateralVec<T>& operator-=(const LateralVec<T>& other) {
        c0 -= other.c0;
        c1 -= other.c1;
        return *this;
    }

    /**
     * Calculate this vector multiplied by scalar \p scale.
     * \param scale scalar
     * \return this vector multiplied by scalar
     */
    template <typename OtherT>
    constexpr auto operator*(const OtherT scale) const -> LateralVec<decltype(c0*scale)> {
PLASK_NO_CONVERSION_WARNING_BEGIN
        return LateralVec<decltype(c0*scale)>(c0 * scale, c1 * scale);
PLASK_NO_WARNING_END
    }

    /**
     * Multiple coordinates of this vector by \p scalar.
     * \param scalar scalar
     * \return *this (after scale)
     */
    LateralVec<T>& operator*=(const T scalar) {
        c0 *= scalar;
        c1 *= scalar;
        return *this;
    }

    /**
     * Calculate this vector divided by scalar \p scale.
     * \param scale scalar
     * \return this vector divided by scalar
     */
    constexpr LateralVec<T> operator/(const T scale) const { return LateralVec<T>(c0 / scale, c1 / scale); }

    /**
     * Divide coordinates of this vector by \p scalar.
     * \param scalar scalar
     * \return *this (after divide)
     */
    LateralVec<T>& operator/=(const T scalar) {
        c0 /= scalar;
        c1 /= scalar;
        return *this;
    }

    /**
     * Calculate vector opposite to this.
     * \return LateralVec<T>(-c0, -c1)
     */
    constexpr LateralVec<T> operator-() const {
        return LateralVec<T>(-c0, -c1);
    }

    /**
     * Square each component of tensor
     * \return squared tensor
     */
    LateralVec<T> sqr() const {
        return LateralVec<T>(c0*c0, c1*c1);
    }

    /**
     * Square each component of tensor in place
     * \return *this (squared)
     */
    LateralVec<T>& sqr_inplace() {
        c0 *= c0; c1 *= c1;
        return *this;
    }

    /**
     * Power of each component of tensor
     * \return squared tensor
     */
    template <typename OtherT>
    LateralVec<T> pow(OtherT a) const {
        return LateralVec<T>(std::pow(c0, a), std::pow(c1, a));
    }

    /**
     * Change i-th coordinate to oposite.
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * \param i number of coordinate
     */
    inline void flip(size_t i) {
        assert(i < 2);
        operator[](i) = -operator[](i);
    }

    /**
     * Get vector similar to this but with changed i-th component to oposite.
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * \param i number of coordinate
     * \return vector similar to this but with changed i-th component to oposite
     */
    inline LateralVec<T> flipped(size_t i) {
        LateralVec<T> res = *this;
        res.flip(i);
        return res;
    }

    /**
     * Print vector to stream using format (where c0 and c1 are vector components): [c0, c1]
     * \param out print destination, output stream
     * \param to_print vector to print
     * \return out stream
     */
    friend inline std::ostream& operator<<(std::ostream& out, const LateralVec<T>& to_print) {
        return out << '[' << to_print.c0 << ", " << to_print.c1 << ']';
    }

    /**
     * A lexical comparison of two vectors, allow to use vector in std::set and std::map as key type.
     *
     * It supports NaN-s (which, due to this method, is greater than all other numbers).
     * \param v vectors to compare
     * \return \c true only if \c this is smaller than the \p v
     */
    template<class OT> inline
    bool operator< (LateralVec< OT> const& v) const {
        if (dbl_compare_lt(this->c0, v.c0)) return true;
        if (dbl_compare_gt(this->c0, v.c0)) return false;
        return dbl_compare_lt(this->c1, v.c1);
    }
};

/**
 * Compute dot product of two vectors \p v1 and \p v2.
 * \param v1 first vector
 * \param v2 second vector
 * \return dot product v1·v2
 */
template <typename T1, typename T2>
inline auto dot(const LateralVec<T1>& v1, const LateralVec<T2>& v2) -> decltype(v1.c0*v2.c0) {
    return ::plask::fma(v1.c0, v2.c0, v1.c1 * v2.c1);	//MSVC needs ::plask::
}

/**
 * Compute (analog of 3d) cross product of two vectors \p v1 and \p v2.
 * \param v1, v2 vectors
 * \return (analog of 3d) cross product of v1 and v2
 */
template <typename T1, typename T2>
inline auto cross(const LateralVec<T1>& v1, const LateralVec<T2>& v2) -> decltype(v1.c0*v2.c1) {
    return ::plask::fma(v1.c0, v2.c1, - v1.c1 * v2.c0);	//MSVC needs ::plask::
}

/** \relates Vec
 * Multiply vector \p v by scalar \p scale.
 * \param scale scalar
 * \param v vector
 * \return vector \p v multiplied by \p scalar
 */
template <typename T, typename OtherT>
inline constexpr auto operator*(const OtherT& scale, const LateralVec<T>& v) -> decltype(v*scale) {
    return v * scale;
}


PLASK_API_EXTERN_TEMPLATE_SPECIALIZATION_STRUCT(LateralVec<double>)
PLASK_API_EXTERN_TEMPLATE_SPECIALIZATION_STRUCT(LateralVec<int>)

} // namespace plask

namespace std {
    template <typename T, typename OtherT>
    plask::LateralVec<T> pow(plask::LateralVec<T> vec, OtherT a) {
        return vec.pow(a);
    }
}

#endif
