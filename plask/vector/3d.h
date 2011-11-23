#ifndef PLASK__VECTOR3D_H
#define PLASK__VECTOR3D_H

#include <cmath>

namespace plask {

/**
 * Vector in 3d space.
 */    
template <typename T>
struct Vector3d {
    
    union {
        ///Allow to access to vector coordinates by index.
        T cordinate[3];
    struct {
        ///Allow to access to vector coordinates by name.
        T x, y, z;
    };
    };
    
    ///Construct uninitialized vector.
    Vector3d() {}
        
    /**
     * Copy constructor from all other 3d vectors.
     * @param p vector to copy from
     */
    template <typename OtherT>
    Vector3d(const Vector3d<OtherT>& p): x(p.x), y(p.y), z(p.z) {}
    
    /**
     * Construct vector with given coordinates.
     * @param x, y, z coordinates
     */
    Vector3d(const T x, const T y, const T z): x(x), y(y), z(z) {}
    
    /**
     * Compare to vectors, this and @a p.
     * @param p vector to compare
     * @return true only if this vector and @a p have equals coordinates
     */
    template <typename OtherT>
    bool operator==(const Vector3d<OtherT>& p) const { return p.x == x && p.y == y && p.z == z; }

    /**
     * Calculate square of vector length.
     * @return square of vector length
     */
    T getLengthSqr() const { return x*x + y*y + z*z; }
        
    /**
     * Calculate vector length.
     * @return vector length
     */
    T getLength() const { return sqrt(getLengthSqr()); }
    
    /**
     * Calculate sum of two vectors, @a this and @a to_add.
     * @param to_add vector to add, can have different data type (than result type will be found using C++ types promotions rules)
     * @return vectors sum
     */
    template <typename OtherT>
    auto operator+(const Vector3d<OtherT>& to_add) -> Vector3d<decltype(x + to_add.x)> const {
        return Vector3d<decltype(this->x + to_add.x)>(x + to_add.x, y + to_add.y, z + to_add.z);
    }
    
    /**
     * Increase coordinates of this vector by coordinates of other vector @a to_add.
     * @param to_add vector to add
     * @return *this (after increase)
     */
    Vector3d<T>& operator+=(const Vector3d<T>& to_add) {
        x += to_add.x;
        y += to_add.y;
        z += to_add.z;
        return *this;
    }
    
    /**
     * Calculate difference of two vectors, @a this and @a to_sub.
     * @param to_sub vector to subtract from this, can have different data type (than result type will be found using C++ types promotions rules)
     * @return vectors difference
     */
    template <typename OtherT>
    auto operator-(const Vector3d<OtherT>& to_sub) -> Vector3d<decltype(x - to_sub.x)> const {
        return Vector3d<decltype(this->x - to_sub.x)>(x - to_sub.x, y - to_sub.y, z - to_sub.z);
    }
    
    /**
     * Decrease coordinates of this vector by coordinates of other vector @a to_sub.
     * @param to_sub vector to subtract
     * @return *this (after decrease)
     */
    Vector3d<T>& operator-=(const Vector3d<T>& to_sub) {
        x -= to_sub.x;
        y -= to_sub.y;
        z -= to_sub.z;
        return *this;
    }

    /**
     * Calculate this vector multiplied by scalar @a scale.
     * @param scale scalar
     * @return this vector multiplied by scalar
     */
    Vector3d<T> operator*(const T scale) const { return Vector3d<T>(x * scale, y * scale, z * scale); }
    
    /**
     * Multiple coordinates of this vector by @a scalar.
     * @param scalar scalar
     * @return *this (after scale)
     */
    Vector3d<T>& operator*=(const T scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }
    
    /**
     * Calculate this vector divided by @a scalar.
     * @param scalar scalar
     * @return this vector divided by @a scalar
     */
    Vector3d<T> operator/(const T scalar) const { return Vector3d<T>(x / scalar, y / scalar, z / scalar); }
    
    /**
     * Divide coordinates of this vector by @a scalar.
     * @param scalar scalar
     * @return *this (after divide)
     */
    Vector3d<T>& operator/=(const T scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }
    
};

}       //namespace plask

#endif