#ifndef PLASK__PRIMITIVES_H
#define PLASK__PRIMITIVES_H

/** @file
This file includes useful geometry primitives, like rectangles, etc.
*/

#include "../vector/2d.h"
#include "../vector/3d.h"

namespace plask {

/**
 * Rectangle class.
 * 
 * Allow for some basic operation on rectangles.
 * Has almost identical interface as Rect3d.
 */
struct Rect2d {

    ///Lower corner of rectangle (with minimal all coordinates).
    Vec2<double> lower;

    ///Upper corner of rectangle (with maximal all coordinates).
    Vec2<double> upper;

    /**
     * Get size of rectangle.
     * @return size of rectangle (its width and height)
     */
    Vec2<double> size() const { return upper - lower; }

    ///Construct uninitilized Rect2d.
    Rect2d() {}

    /**
     * Construct rectangle.
     * @param lower lower corner of rectangle (with minimal all coordinates)
     * @param upper upper corner of rectangle (with maximal all coordinates)
     */
    Rect2d(const Vec2<double>& lower, const Vec2<double>& upper): lower(lower), upper(upper) {}
    
    /**
     * Compare two rectangles, this and @a r.
     * @param r rectangle to compare
     * @return true only if this rectangle and @a p have equals coordinates
     */
    bool operator==(const Rect2d& r) const;
    
    /**
     * Compare two rectangles, this and @a r.
     * @param r rectangle to compare
     * @return true only if this rectangle and @a p don't have equals coordinates
     */
    bool operator!=(const Rect2d& r) const;
    
    /**
     * Ensure that: lower.x <= upper.x and lower.y <= upper.y.
     * Change x or y of lower and upper if necessary.
     */
    void fix();

    /**
     * Check if point is inside rectangle.
     * @param p point
     * @return true only if point is inside this rectangle
     */
    bool inside(const Vec2<double>& p) const;

    /**
     * Check if this and other rectangles have common points.
     * @param other rectangle
     * @return true only if this and other have common points
     */
    bool intersect(const Rect2d& other) const;

    /**
     * Make this rectangle, the minimal rectangle which include this and given point @a p.
     * @param p point which should be inside rectangle
     */
    void include(const Vec2<double>& p);

    /**
     * Make this rectangle, the minimal rectangle which include this and @a other rectangle.
     * @param other point which should be inside rectangle
     */
    void include(const Rect2d& other);

    Rect2d translated(const Vec2<double>& translation_vec) const { return Rect2d(lower + translation_vec, upper + translation_vec); }
    
    void translate(const Vec2<double>& translation_vec) { lower += translation_vec; upper += translation_vec; }
    
    /**
     * Print rectangle to stream.
     * @param out print destination, output stream
     * @param to_print rectangle to print
     * @return out stream
     */
    friend inline std::ostream& operator<<(std::ostream& out, const Rect2d& to_print) {
        return out << '[' << to_print.lower << ", " << to_print.upper << ']';
    }
    
};

/**
 * Cuboid class.
 *
 * Allow for some basic operation on cuboid.
 * Has almost identical interface as Rect2d.
 */
struct Rect3d {

    ///Position of lower corner of cuboid (with minimal all coordinates).
    Vec3<double> lower;

    ///Position of upper corner of cuboid (with maximal all coordinates).
    Vec3<double> upper;

    Vec3<double> size() const { return upper - lower; }

    ///Construct uninitilized Rect2d.
    Rect3d() {}

    Rect3d(const Vec3<double>& lower, const Vec3<double>& upper): lower(lower), upper(upper) {}
    
    /**
     * Compare two rectangles, this and @a r.
     * @param r rectangle to compare
     * @return true only if this rectangle and @a p have equals coordinates
     */
    bool operator==(const Rect3d& r) const;
    
    /**
     * Compare two rectangles, this and @a r.
     * @param r rectangle to compare
     * @return true only if this rectangle and @a p don't have equals coordinates
     */
    bool operator!=(const Rect3d& r) const;
    
    /**
     * Ensure that: lower.x <= upper.x and lower.y <= upper.y.
     * Change x or y of lower and upper if necessary.
     */
    void fix();

    /**
     * Check if point is inside rectangle.
     * @param p point
     * @return true only if point is inside this rectangle
     */
    bool inside(const Vec3<double>& p) const;

    /**
     * Check if this and other rectangles have common points.
     * @param other rectangle
     * @return true only if this and other have common points
     */
    bool intersect(const Rect3d& other) const;

    /**
     * Make this rectangle, the minimal rectangle which include this and given point @a p.
     * @param p point which should be inside rectangle
     */
    void include(const Vec3<double>& p);

    /**
     * Make this rectangle, the minimal rectangle which include this and @a other rectangle.
     * @param other point which should be inside rectangle
     */
    void include(const Rect3d& other);

    Rect3d translated(const Vec3<double>& translation_vec) const { return Rect3d(lower + translation_vec, upper + translation_vec); }
    
    void translate(const Vec3<double>& translation_vec) { lower += translation_vec; upper += translation_vec; }
    
    /**
     * Print rectangle to stream.
     * @param out print destination, output stream
     * @param to_print rectangle to print
     * @return out stream
     */
    friend inline std::ostream& operator<<(std::ostream& out, const Rect3d& to_print) {
        return out << '[' << to_print.lower << ", " << to_print.upper << ']';
    }
    
};

/**
 * Typedefs for primitives for given space dimensions.
 */
template <int dim>
struct Primitive {};

template <>
struct Primitive<2> {
    typedef Rect2d Rect;
    typedef Vec2<double> Vec;
    static const int dim = 2;
    
    static const Vec ZERO_VEC;
};

template <>
struct Primitive<3> {
    typedef Rect3d Rect;
    typedef Vec3<double> Vec;
    static const int dim = 3;
    
    static const Vec ZERO_VEC;
};

} // namespace plask

#endif

