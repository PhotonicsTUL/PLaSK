#ifndef PLASK__PRIMITIVES_H
#define PLASK__PRIMITIVES_H

#include "../vector/2d.h"
#include "../vector/3d.h"

namespace plask {

struct Rect2d {
    
    Vec2<double> lower;
        
    Vec2<double> upper;
        
    Vec2<double> size() const { return upper - lower; }
    
    Rect2d(const Vec2<double>& lower, const Vec2<double>& upper): lower(lower), upper(upper) {}
    
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
    
};

struct Rect3d {
    
    Vec3<double> lower;
        
    Vec3<double> upper;
        
    Vec3<double> size() const { return upper - lower; }
    
    Rect3d(const Vec3<double>& lower, const Vec3<double>& upper): lower(lower), upper(upper) {}
    
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

}       // namespace plask

#endif

