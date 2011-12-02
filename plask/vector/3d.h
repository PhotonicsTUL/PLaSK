#ifndef PLASK__VECTOR3D_H
#define PLASK__VECTOR3D_H

#include <cmath>

namespace plask {

/**
 * Vector in 3d space.
 */
template <typename T>
struct Vec3 {

    union {
        /// Allow to access to vector coordinates by index.
        T coordinate[3];
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
     * Compare to vectors, this and @a p.
     * @param p vector to compare
     * @return true only if this vector and @a p have equals coordinates
     */
    template <typename OtherT>
    bool operator==(const Vec3<OtherT>& p) const { return p.c0 == c0 && p.c1 == c1 && p.c2 == c2; }

    /**
     * Calculate square of vector length.
     * @return square of vector length
     */
    T lengthSqr() const { return c0*c0 + c1*c1 + c2*c2; }

    /**
     * Calculate vector length.
     * @return vector length
     */
    T length() const { return sqrt(lengthSqr()); }

    /**
     * Calculate sum of two vectors, @a this and @a to_add.
     * @param to_add vector to add, can have different data type (than result type will be found using C++ types promotions rules)
     * @return vectors sum
     */
    template <typename OtherT>
    auto operator+(const Vec3<OtherT>& to_add) -> Vec3<decltype(c0 + to_add.c0)> const {
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
    auto operator-(const Vec3<OtherT>& to_sub) -> Vec3<decltype(c0 - to_sub.c0)> const {
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

};

/**
 * Multiple vector @a v by scalar @a scale.
 * @param scale scalar
 * @param v vector
 * @return vector multiplied by scalar
 */
template <typename T>
inline Vec3<T> operator*(const T scale, const Vec3<T>& v) { return v*scale; }


} //namespace plask

#endif
