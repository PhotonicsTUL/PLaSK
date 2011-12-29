#ifndef PLASK__VECTOR3D_H
#define PLASK__VECTOR3D_H

#include <iostream>
#include <config.h>

namespace plask {

/**
 * Vector in 3d space.
 */
template <typename T = double>

struct Vec3 {
    union {
        /// Allow to access to vector coordinates by index.
        T components[3];
        struct {
            /// Allow to access to vector coordinates by name.
            T c0, c1, c2;
        };
        struct {
            T x, y, z;
        };
        struct {
            T a, b, c;
        };
        struct {
            T r, phi;
        };
    };

    /// Construct uninitialized vector.
    Vec3() {}

    /**
     * Copy constructor from all other 3d vectors.
     * @param p vector to copy from
     */
    template <typename OtherT>
    Vec3(const Vec3<OtherT>& p): c0(p.c0), c1(p.c1), c2(p.c2) {}

    /**
     * Construct vector with given coordinates.
     * @param c0, c1, c2 coordinates
     */
    Vec3(const T c0, const T c1, const T c2): c0(c0), c1(c1), c2(c2) {}

    /**
     * Compare two vectors, this and @a p.
     * @param p vector to compare
     * @return true only if this vector and @a p have equals coordinates
     */
    template <typename OtherT>
    bool operator==(const Vec3<OtherT>& p) const { return p.c0 == c0 && p.c1 == c1 && p.c2 == c2; }

    /**
     * Compare two vectors, this and @a p.
     * @param p vector to compare
     * @return true only if this vector and @a p don't have equals coordinates
     */
    template <typename OtherT>
    bool operator!=(const Vec3<OtherT>& p) const { return p.c0 != c0 || p.c1 != c1 || p.c2 != c2; }

    /**
     * Get i-th component
     * WARNING This function does not check if param is valid (for efficiency reasons)
     * @param number of coordinate
     * @return i-th component
     */
    inline T& operator[](size_t i) {
        return components[i];
    }

    /**
     * Get i-th component
     * WARNING This function does not check if param is valid (for efficiency reasons)
     * @param number of coordinate
     * @return i-th component
     */
    inline T operator[](size_t i) const {
        return components[i];
    }

    /**
     * Calculate square of vector magnitude.
     * @return square of vector magnitude
     */
    inline T magnitude2() const { return c0*c0 + c1*c1 + c2*c2; }

    /**
     * Calculate vector magnitude.
     * @return vector magnitude
     */
    inline T magnitude() const { return sqrt(magnitude2()); }

    /**
     * Calculate sum of two vectors, @a this and @a to_add.
     * @param to_add vector to add, can have different data type (than result type will be found using C++ types promotions rules)
     * @return vectors sum
     */
    template <typename OtherT>
    auto operator+(const Vec3<OtherT>& to_add) const -> Vec3<decltype(c0 + to_add.c0)> {
        return Vec3<decltype(this->c0 + to_add.c0)>(c0 + to_add.c0, c1 + to_add.c1, c2 + to_add.c2);
    }

    /**
     * Increase coordinates of this vector by coordinates of other vector @a to_add.
     * @param to_add vector to add
     * @return *this (after increase)
     */
    Vec3<T>& operator+=(const Vec3<T>& to_add) {
        c0 += to_add.c0;
        c1 += to_add.c1;
        c2 += to_add.c2;
        return *this;
    }

    /**
     * Calculate difference of two vectors, @a this and @a to_sub.
     * @param to_sub vector to subtract from this, can have different data type (than result type will be found using C++ types promotions rules)
     * @return vectors difference
     */
    template <typename OtherT>
    auto operator-(const Vec3<OtherT>& to_sub) const -> Vec3<decltype(c0 - to_sub.c0)> {
        return Vec3<decltype(this->c0 - to_sub.c0)>(c0 - to_sub.c0, c1 - to_sub.c1, c2 - to_sub.c2);
    }

    /**
     * Decrease coordinates of this vector by coordinates of other vector @a to_sub.
     * @param to_sub vector to subtract
     * @return *this (after decrease)
     */
    Vec3<T>& operator-=(const Vec3<T>& to_sub) {
        c0 -= to_sub.c0;
        c1 -= to_sub.c1;
        c2 -= to_sub.c2;
        return *this;
    }

    /**
     * Calculate this vector multiplied by scalar @a scale.
     * @param scale scalar
     * @return this vector multiplied by scalar
     */
    Vec3<T> operator*(const T scale) const { return Vec3<T>(c0 * scale, c1 * scale, c2 * scale); }

    /**
     * Multiple coordinates of this vector by @a scalar.
     * @param scalar scalar
     * @return *this (after scale)
     */
    Vec3<T>& operator*=(const T scalar) {
        c0 *= scalar;
        c1 *= scalar;
        c2 *= scalar;
        return *this;
    }

    /**
     * Calculate this vector divided by @a scalar.
     * @param scalar scalar
     * @return this vector divided by @a scalar
     */
    Vec3<T> operator/(const T scalar) const { return Vec3<T>(c0 / scalar, c1 / scalar, c2 / scalar); }

    /**
     * Divide coordinates of this vector by @a scalar.
     * @param scalar scalar
     * @return *this (after divide)
     */
    Vec3<T>& operator/=(const T scalar) {
        c0 /= scalar;
        c1 /= scalar;
        c2 /= scalar;
        return *this;
    }

    /**
     * Calculate vector opposite to this.
     * @return Vec3<T>(-c0, -c1, -c2)
     */
    Vec3<T> operator-() const {
        return Vec3<T>(-c0, -c1, -c2);
    }

    /**
     * Print vector to stream using format (where c0, c1 and c2 are vector coordinates): [c0, c1, c2]
     * @param out print destination, output stream
     * @param to_print vector to print
     * @return out stream
     */
    friend inline std::ostream& operator<<(std::ostream& out, const Vec3<T>& to_print) {
        return out << '[' << to_print.c0 << ", " << to_print.c1 << ", " << to_print.c2 << ']';
    }

};

/**
 * Multiple vector @a v by scalar @a scale.
 * @param scale scalar
 * @param v vector
 * @return vector multiplied by scalar
 */
template <typename T>
inline Vec3<T> operator*(const T scale, const Vec3<T>& v) { return v*scale; }

/**
 * Calculate square of vector magnitude.
 * @param v a vector
 * @return square of vector magnitude
 */
template <typename T>
T abs2(const Vec3<T>& v) { return v.magnitude2(); }

/**
 * Calculate vector magnitude.
 * @param v a vector
 * @return vector magnitude
 */
template <typename T>
T abs(const Vec3<T>& v) { return v.magnitude(); }

/**
 * Calculate vector conjugate.
 * @param v a vector
 * @return conjugate vector
 */
template <typename T>
inline Vec3<T> conj(const Vec3<T>& v) { return Vec3<T> {conj(v.c0), conj(v.c1), conj(v.c2)}; }

/**
 * Compute dot product of two vectors @a v1 and @a v2
 * @param v1 first vector
 * @param v2 second vector
 * @return dot product v1·v2
 */
template <typename T1, typename T2>
inline auto dot(const Vec3<T1>& v1, const Vec3<T2>& v2) -> decltype(v1.c0*v2.c0) {
    return v1.c0 * v2.c0 + v1.c1 * v2.c1 + v1.c2 * v2.c2;
}

/**
 * Helper to create 3d vector.
 * @param c0, c1, c2 vector coordinates.
 */
template <typename T>
inline Vec3<T> vec(const T c0, const T c1, const T c2) {
    return Vec3<T>(c0, c1, c2);
}

} //namespace plask

#endif
