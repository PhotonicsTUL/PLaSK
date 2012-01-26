#ifndef PLASK__PRIMITIVES_H
#define PLASK__PRIMITIVES_H

/** @file
This file includes useful geometry primitives, like rectangles, etc.
*/

#include "../vec.h"

namespace plask {

/**
 * Rectangle class.
 *
 * Allows for some basic operation on rectangles.
 * Has almost identical interface as Rect3d.
 */
struct Rect2d {

    ///Lower corner of rectangle (with minimal all coordinates).
    Vec<2,double> lower;

    ///Upper corner of rectangle (with maximal all coordinates).
    Vec<2,double> upper;

    /**
     * Get size of rectangle.
     * @return size of rectangle (its width and height)
     */
    Vec<2,double> size() const { return upper - lower; }
    
    /**
     * Calculate size of this in up direction.
     * @return upper.up - lower.up
     */
    double sizeUp() const { return upper.up - lower.up; }

    ///Construct uninitialized Rect2d.
    Rect2d() {}

    /**
     * Construct rectangle.
     * @param lower lower corner of rectangle (with minimal all coordinates)
     * @param upper upper corner of rectangle (with maximal all coordinates)
     */
    Rect2d(const Vec<2,double>& lower, const Vec<2,double>& upper): lower(lower), upper(upper) {}

    /**
     * Compare two rectangles, @c this and @p r.
     * @param r rectangle to compare
     * @return true only if @c this rectangle and @p p have equals coordinates
     */
    bool operator==(const Rect2d& r) const;

    /**
     * Compare two rectangles, @c this and @p r.
     * @param r rectangle to compare
     * @return @c true only if @c this rectangle and @p p don't have equals coordinates
     */
    bool operator!=(const Rect2d& r) const;

    /**
     * Ensure that lower[0] <= upper[0] and lower[1] <= upper[1].
     * Exchange x or y of lower and upper if necessary.
     */
    void fix();

    /**
     * Check if the point is inside rectangle.
     * @param p point
     * @return true only if point is inside this rectangle
     */
    bool inside(const Vec<2,double>& p) const;

    /**
     * Check if this and other rectangles have common points.
     * @param other rectangle
     * @return true only if this and other have common points
     */
    bool intersect(const Rect2d& other) const;

    /**
     * Make this rectangle, the minimal one which include @c this and given point @p p.
     * @param p point which should be inside rectangle
     */
    void include(const Vec<2,double>& p);

    /**
     * Make this rectangle, the minimal one which include @c this and @p other rectangle.
     * @param other point which should be inside rectangle
     */
    void include(const Rect2d& other);

    /**
     * Get translated copy of this.
     * @param translation_vec translation vector
     * @return this translated by @p translation_vec
     */
    Rect2d translated(const Vec<2,double>& translation_vec) const { return Rect2d(lower + translation_vec, upper + translation_vec); }
    
    /**
     * Get translated copy of this.
     * @param trasnalation_in_up_dir translation in up direction
     * @return this translated up by @p trasnalation_in_up_dir
     */
    Rect2d translatedUp(const double trasnalation_in_up_dir) const { return translated(vec(0.0, trasnalation_in_up_dir)); }
    
    /**
     * Translate this by @p translation_vec.
     * @param translation_vec translation vector
     */
    void translate(const Vec<2,double>& translation_vec) { lower += translation_vec; upper += translation_vec; }
    
    /**
     * Translate this up by @p trasnalation_in_up_dir.
     * @param trasnalation_in_up_dir translation in up direction
     */
    void translateUp(const double trasnalation_in_up_dir) { lower.up += trasnalation_in_up_dir; upper.up += trasnalation_in_up_dir; }

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
    Vec<3,double> lower;

    ///Position of upper corner of cuboid (with maximal all coordinates).
    Vec<3,double> upper;

    /**
     * Calculate size of this.
     * @return upper - lower
     */
    Vec<3,double> size() const { return upper - lower; }
    
    /**
     * Calculate size of this in up direction.
     * @return upper.up - lower.up
     */
    double sizeUp() const { return upper.up - lower.up; }

    ///Construct uninitialized Rect3d.
    Rect3d() {}

    /**
     * Construct Rect3d with given lower and upper corner positions.
     * @param lower position of lower corner of cuboid (with minimal all coordinates)
     * @param upper position of upper corner of cuboid (with maximal all coordinates)
     */
    Rect3d(const Vec<3,double>& lower, const Vec<3,double>& upper): lower(lower), upper(upper) {}

    /**
     * Compare two rectangles, @c this and @p r.
     * @param r rectangle to compare
     * @return true only if @c this rectangle and @p p have equals coordinates
     */
    bool operator==(const Rect3d& r) const;

    /**
     * Compare two rectangles, @c this and @p r.
     * @param r rectangle to compare
     * @return true only if @c this rectangle and @p p don't have equals coordinates
     */
    bool operator!=(const Rect3d& r) const;

    /**
     * Ensure that lower[0] <= upper.c0, lower[1] <= upper[1], and lower[2] <= upper[3].
     * Excange components of lower and upper if necessary.
     */
    void fix();

    /**
     * Check if point is inside rectangle.
     * @param p point
     * @return true only if point is inside this rectangle
     */
    bool inside(const Vec<3,double>& p) const;

    /**
     * Check if this and other rectangles have common points.
     * @param other rectangle
     * @return true only if this and other have common points
     */
    bool intersect(const Rect3d& other) const;

    /**
     * Make this rectangle, the minimal one which include @c this and given point @p p.
     * @param p point which should be inside rectangle
     */
    void include(const Vec<3,double>& p);

    /**
     * Make this rectangle, the minimal one which include @c this and @p other rectangle.
     * @param other point which should be inside rectangle
     */
    void include(const Rect3d& other);

    Rect3d translated(const Vec<3,double>& translation_vec) const { return Rect3d(lower + translation_vec, upper + translation_vec); }

    /**
     * Get translated copy of this.
     * @param trasnalation_in_up_dir translation in up direction
     * @return @c this translated up by @p trasnalation_in_up_dir
     */
    Rect3d translatedUp(const double trasnalation_in_up_dir) const { Rect3d r = *this; r.translateUp(trasnalation_in_up_dir); return r; }
    
    void translate(const Vec<3,double>& translation_vec) { lower += translation_vec; upper += translation_vec; }
    
    /**
     * Translate this up by @p trasnalation_in_up_dir.
     * @param trasnalation_in_up_dir translation in up direction
     */
    void translateUp(const double trasnalation_in_up_dir) { lower.up += trasnalation_in_up_dir; upper.up += trasnalation_in_up_dir; }

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
 * Define types of primitives and constants in space with given number of dimensions.
 * @tparam dim number of dimensions, 2 or 3
 */
template <int dim>
struct Primitive {};

/**
 * Specialization of Primitive, which define types of primitives and constants in space with 1 dimensions.
 */
template <>
struct Primitive<1> {

    /// Real (double) vector type in 1d space.
    typedef double DVec;

    /// Number of dimensions (1).
    static const int dim = 1;

    /// Zeroed 1d vector.
    static const DVec ZERO_VEC;
};

/**
 * Specialization of Primitive, which define types of primitives and constants in space with 2 dimensions.
 */
template <>
struct Primitive<2> {

    /// Rectangle type in 2d space.
    typedef Rect2d Rect;

    /// Real (double) vector type in 2d space.
    typedef Vec<2,double> DVec;

    /// Number of dimensions (2).
    static const int dim = 2;

    /// Zeroed 2d vector.
    static const DVec ZERO_VEC;
};

/**
 * Specialization of Primitive, which define types of primitives and constants in space with 3 dimensions.
 */
template <>
struct Primitive<3> {

    /// Rectangle type (cuboid) in 3d space.
    typedef Rect3d Rect;

    /// Real (double) vector type in 3d space.
    typedef Vec<3,double> DVec;

    /// Number of dimensions (3).
    static const int dim = 3;

    /// Zeroed 3d vector.
    static const DVec ZERO_VEC;
};

} // namespace plask

#endif

