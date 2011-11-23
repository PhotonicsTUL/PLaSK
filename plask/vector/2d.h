#ifndef PLASK__VECTOR2D_H
#define PLASK__VECTOR2D_H

#include <cmath>

namespace plask {

/**
 * Vector in 2d space.
 */    
template <typename T>
struct Vector2d {
    
    union {
        ///Allow to access to vector coordinates by index.
        T cordinate[2];
    struct {
        ///Allow to access to vector coordinates by name.
        T x, y;
    };
    };
    
    ///Construct uninitialized vector.
    Vector2d() {}
        
    /**
     * Copy constructor from all other 2d vectors.
     * @param p vector to copy from
     */
    template <typename OtherT>
    Vector2d(const Vector2d<OtherT>& p): x(p.x), y(p.y) {}
    
    /**
     * Construct vector with given coordinates.
     * @param x, y coordinates
     */
    Vector2d(const T x, const T y): x(x), y(y) {}
    
    /**
     * Compare to vectors, this and @a p.
     * @param p vector to compare
     * @return true only if this vector and @a p have equals coordinates
     */
    template <typename OtherT>
    bool operator==(const Vector2d<OtherT>& p) const { return p.x == x && p.y == y; }

    /**
     * Calculate square of vector length.
     * @return square of vector length
     */
    T getLengthSqr() const { return x*x + y*y; }
        
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
    auto operator+(const Vector2d<OtherT>& to_add) -> Vector2d<decltype(x + to_add.x)> const {
        return Vector2d<decltype(this->x + to_add.x)>(x + to_add.x, y + to_add.y);
    }
    
    /**
     * Increase coordinates of this vector by coordinates of other vector @a to_add.
     * @param to_add vector to add
     * @return *this (after increase)
     */
    Vector2d<T>& operator+=(const Vector2d<T>& to_add) {
        x += to_add.x;
        y += to_add.y;
        return *this;
    }
    
    /**
     * Calculate difference of two vectors, @a this and @a to_sub.
     * @param to_sub vector to subtract from this, can have different data type (than result type will be found using C++ types promotions rules)
     * @return vectors difference
     */
    template <typename OtherT>
    auto operator-(const Vector2d<OtherT>& to_sub) -> Vector2d<decltype(x - to_sub.x)> const {
        return Vector2d<decltype(this->x - to_sub.x)>(x - to_sub.x, y - to_sub.y);
    }
    
    /**
     * Decrease coordinates of this vector by coordinates of other vector @a to_sub.
     * @param to_sub vector to subtract
     * @return *this (after decrease)
     */
    Vector2d<T>& operator-=(const Vector2d<T>& to_sub) {
        x -= to_sub.x;
        y -= to_sub.y;
        return *this;
    }

    /**
     * Calculate this vector multiplied by scalar @a scale.
     * @param scale scalar
     * @return this vector multiplied by scalar
     */
    Vector2d<T> operator*(const T scale) const { return Vector2d<T>(x * scale, y * scale); }
    
    /**
     * Multiple coordinates of this vector by @a scalar.
     * @param scalar scalar
     * @return *this (after scale)
     */
    Vector2d<T>& operator*=(const T scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }
    
    /**
     * Calculate this vector divided by scalar @a scale.
     * @param scale scalar
     * @return this vector divided by scalar
     */
    Vector2d<T> operator/(const T scale) const { return Vector2d<T>(x / scale, y / scale); }
    
    /**
     * Divide coordinates of this vector by @a scalar.
     * @param scalar scalar
     * @return *this (after divide)
     */
    Vector2d<T>& operator/=(const T scalar) {
        x /= scalar;
        y /= scalar;
        return *this;
    }
    
};

}       //namespace plask

#endif