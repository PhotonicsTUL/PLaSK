#ifndef PLASK__PRIMITIVES_H
#define PLASK__PRIMITIVES_H

/** @file
This file includes useful geometry primitives, like boxes, etc.
*/

#include "../vec.h"
#include "../exceptions.h"

namespace plask {

/**
 * Rectangle class.
 *
 * Allows for some basic operation on boxes.
 * Has almost identical interface as .
 */
struct Box2D {

    ///Lower corner of box (with minimal all coordinates).
    Vec<2,double> lower;

    ///Upper corner of box (with maximal all coordinates).
    Vec<2,double> upper;

    /**
     * Get size of box.
     * @return size of box (its width and height)
     */
    Vec<2,double> size() const { return upper - lower; }

    /**
     * Calculate height of this.
     * @return upper.vert() - lower.up
     */
    double height() const { return upper.vert() - lower.vert(); }

    ///Construct uninitialized .
    Box2D() {}

    /**
     * Construct box.
     * @param lower lower corner of box (with minimal all coordinates)
     * @param upper upper corner of box (with maximal all coordinates)
     */
    Box2D(const Vec<2,double>& lower, const Vec<2,double>& upper): lower(lower), upper(upper) {}

    /**
     * Construct box.
     * @param x_lo, y_lo lower corner of box (with minimal all coordinates)
     * @param x_up, y_up upper corner of box (with maximal all coordinates)
     */
    Box2D(double x_lo, double y_lo, double x_up, double y_up): lower(x_lo, y_lo), upper(x_up, y_up) {}

    static Box2D invalidInstance() {
        Box2D r; r.makeInvalid(); return r;
    }

    /**
     * Compare two boxes, @c this and @p r.
     * @param r box to compare
     * @return true only if @c this box and @p p have equals coordinates
     */
    bool operator==(const Box2D& r) const;

    /**
     * Compare two boxes, @c this and @p r.
     * @param r box to compare
     * @return @c true only if @c this box and @p p don't have equals coordinates
     */
    bool operator!=(const Box2D& r) const;

    /**
     * Ensure that lower[0] <= upper[0] and lower[1] <= upper[1].
     * Exchange x or y of lower and upper if necessary.
     */
    void fix();

    /**
     * Check if the point is inside the box.
     * @param p point
     * @return true only if point is inside this box
     */
    bool includes(const Vec<2,double>& p) const;

    /**
     * Check if this and other boxes have common points.
     * @param other box
     * @return true only if this and other have common points
     */
    bool intersects(const Box2D& other) const;

    /**
     * Make this box, the minimal one which include @c this and given point @p p.
     * @param p point which should be inside box
     */
    void makeInclude(const Vec<2,double>& p);

    /**
     * Make this box, the minimal one which include @c this and @p other box.
     * @param other point which should be inside box
     */
    void makeInclude(const Box2D& other);

    /**
     * Get translated copy of this.
     * @param translation_vec translation vector
     * @return this translated by @p translation_vec
     */
    Box2D translated(const Vec<2,double>& translation_vec) const { return Box2D(lower + translation_vec, upper + translation_vec); }

    /**
     * Get translated copy of this.
     * @param translation_vec translation vector
     * @return this translated by @p translation_vec
     */
    Box2D operator+(const Vec<2,double>& translation_vec) const { return Box2D(lower + translation_vec, upper + translation_vec); }

    /**
     * Get translated copy of this.
     * @param translation_vec translation vector
     * @return this translated by @p translation_vec
     */
    Box2D operator-(const Vec<2,double>& translation_vec) const { return Box2D(lower - translation_vec, upper - translation_vec); }

    /**
     * Get translated copy of this.
     * @param trasnalation_in_up_dir translation in up direction
     * @return this translated up by @p trasnalation_in_up_dir
     */
    Box2D translatedUp(const double trasnalation_in_up_dir) const { return translated(vec(0.0, trasnalation_in_up_dir)); }

    /**
     * Translate this by @p translation_vec.
     * @param translation_vec translation vector
     */
    void translate(const Vec<2,double>& translation_vec) { lower += translation_vec; upper += translation_vec; }

    /**
     * Translate this by @p translation_vec.
     * @param translation_vec translation vector
     */
    Box2D& operator+=(const Vec<2,double>& translation_vec) { lower += translation_vec; upper -= translation_vec; return *this; }

    /**
     * Translate this by @p translation_vec.
     * @param translation_vec translation vector
     */
    Box2D& operator-=(const Vec<2,double>& translation_vec) { lower -= translation_vec; upper -= translation_vec; return *this; }

    /**
     * Translate this up by @p trasnalation_in_up_dir.
     * @param trasnalation_in_up_dir translation in up direction
     */
    void translateUp(const double trasnalation_in_up_dir) { lower.vert() += trasnalation_in_up_dir; upper.vert() += trasnalation_in_up_dir; }

    /**
     * Translate a point to be inside the box by shifting to the closest edge.
     * This method assumes that the box is fixed.
     * @param p given point
     * @return point shifted to the boxes
     */
    Vec<2,double> moveInside(Vec<2,double> p) const;

    /**
     * Print box to stream.
     * @param out print destination, output stream
     * @param to_print box to print
     * @return out stream
     */
    friend inline std::ostream& operator<<(std::ostream& out, const Box2D& to_print) {
        return out << '[' << to_print.lower << ", " << to_print.upper << ']';
    }

    /**
     * Check if this box is valid.
     *
     * Valid box has: upper.c0 >= lower.c0 && upper.c1 >= lower.c1
     * @return @c true only if this box is valid
     */
    bool isValid() const { return upper.c0 >= lower.c0 && upper.c1 >= lower.c1; }

    /**
     * Check if this box is empty.
     *
     * Empty box has: lower == upper
     * @return @c true only if this box is empty
     */
    bool isEmpty() const { return lower == upper; }

    /**
     * Set this box coordinates to invalid once, so isValid() returns @c false after this call.
     * @see isValid()
     */
    void makeInvalid() { lower = vec(0.0, 0.0); upper = vec(-1.0, -1.0); }

    /**
     * Calculate area of the box.
     * @return area of the box
     */
    double getArea() const {
        Vec<2, double> v = size();
        return v.c0 * v.c1;
    }

    /**
     * Change i-th coordinate to oposite (mirror).
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * @param flipDir number of coordinate
     */
    inline void flip(size_t flipDir) {
        assert(flipDir < 2);
        double temp = lower[flipDir];
        lower[flipDir] = - upper[flipDir];
        upper[flipDir] = - temp;
    }

    /**
     * Get vector similar to this but with changed i-th coordinate to oposite.
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * @param flipDir number of coordinate
     * @return box similar to this but with mirrored i-th coordinate
     */
    inline Box2D fliped(size_t i) {
        Box2D res = *this;
        res.flip(i);
        return res;
    }
};

/**
 * Cuboid class.
 *
 * Allow for some basic operation on cuboid.
 * Has almost identical interface as .
 */
struct Box3D {

    /// Position of lower corner of cuboid (with minimal all coordinates).
    Vec<3,double> lower;

    /// Position of upper corner of cuboid (with maximal all coordinates).
    Vec<3,double> upper;

    /**
     * Calculate size of this.
     * @return upper - lower
     */
    Vec<3,double> size() const { return upper - lower; }

    /**
     * Calculate size of this in up direction.
     * @return upper.vert() - lower.up
     */
    double height() const { return upper.vert() - lower.vert(); }

    /// Construct uninitialized .
    Box3D() {}

    /**
     * Construct  with given lower and upper corner positions.
     * @param lower position of lower corner of cuboid (with minimal all coordinates)
     * @param upper position of upper corner of cuboid (with maximal all coordinates)
     */
    Box3D(const Vec<3,double>& lower, const Vec<3,double>& upper): lower(lower), upper(upper) {}

    /**
     * Construct box.
     * @param x_lo, y_lo, z_lo lower corner of box (with minimal all coordinates)
     * @param x_up, y_up, z_up upper corner of box (with maximal all coordinates)
     */
    Box3D(double x_lo, double y_lo, double z_lo, double x_up, double y_up, double z_up): lower(x_lo, y_lo, z_lo), upper(x_up, y_up, z_up) {}

    static Box3D invalidInstance() {
        Box3D r; r.makeInvalid(); return r;
    }

    /**
     * Compare two boxes, @c this and @p r.
     * @param r box to compare
     * @return true only if @c this box and @p p have equals coordinates
     */
    bool operator==(const Box3D& r) const;

    /**
     * Compare two boxes, @c this and @p r.
     * @param r box to compare
     * @return true only if @c this box and @p p don't have equals coordinates
     */
    bool operator!=(const Box3D& r) const;

    /**
     * Ensure that lower[0] <= upper.c0, lower[1] <= upper[1], and lower[2] <= upper[3].
     * Excange components of lower and upper if necessary.
     */
    void fix();

    /**
     * Check if point is inside box.
     * @param p point
     * @return true only if point is inside this box
     */
    bool includes(const Vec<3,double>& p) const;

    /**
     * Check if this and other boxes have common points.
     * @param other box
     * @return true only if this and other have common points
     */
    bool intersects(const Box3D& other) const;

    /**
     * Make this box, the minimal one which include @c this and given point @p p.
     * @param p point which should be inside box
     */
    void makeInclude(const Vec<3,double>& p);

    /**
     * Make this box, the minimal one which include @c this and @p other box.
     * @param other point which should be inside box
     */
    void makeInclude(const Box3D& other);

    /**
     * Get translated copy of this.
     * @param translation_vec translation vector
     * @return this translated by @p translation_vec
     */
    Box3D translated(const Vec<3,double>& translation_vec) const { return Box3D(lower + translation_vec, upper + translation_vec); }

    /**
     * Get translated copy of this.
     * @param translation_vec translation vector
     * @return this translated by @p translation_vec
     */
    Box3D operator+(const Vec<3,double>& translation_vec) const { return Box3D(lower + translation_vec, upper + translation_vec); }

    /**
     * Get translated copy of this.
     * @param translation_vec translation vector
     * @return this translated by @p translation_vec
     */
    Box3D operator-(const Vec<3,double>& translation_vec) const { return Box3D(lower - translation_vec, upper - translation_vec); }

    /**
     * Get translated copy of this.
     * @param trasnalation_in_up_dir translation in up direction
     * @return @c this translated up by @p trasnalation_in_up_dir
     */
    Box3D translatedUp(const double trasnalation_in_up_dir) const {
        Box3D r = *this; r.translateUp(trasnalation_in_up_dir); return r; }

    /**
     * Translate this by @p translation_vec.
     * @param translation_vec translation vector
     */
    void translate(const Vec<3,double>& translation_vec) { lower += translation_vec; upper += translation_vec; }

    /**
     * Translate this by @p translation_vec.
     * @param translation_vec translation vector
     */
    Box3D& operator+=(const Vec<3,double>& translation_vec) { lower += translation_vec; upper -= translation_vec; return *this; }

    /**
     * Translate this by @p translation_vec.
     * @param translation_vec translation vector
     */
    Box3D& operator-=(const Vec<3,double>& translation_vec) { lower -= translation_vec; upper -= translation_vec; return *this; }

    /**
     * Translate this up by @p trasnalation_in_up_dir.
     * @param trasnalation_in_up_dir translation in up direction
     */
    void translateUp(const double trasnalation_in_up_dir) { lower.vert() += trasnalation_in_up_dir; upper.vert() += trasnalation_in_up_dir; }

    /**
     * Translate a point to be inside the box by shifting to the closest edge.
     * This method assumes that the box is fixed.
     * @param p given point
     * @return point shifted to the boxes
     */
    Vec<3,double> moveInside(Vec<3,double> p) const;

    /**
     * Print box to stream.
     * @param out print destination, output stream
     * @param to_print box to print
     * @return out stream
     */
    friend inline std::ostream& operator<<(std::ostream& out, const Box3D& to_print) {
        return out << '[' << to_print.lower << ", " << to_print.upper << ']';
    }

    /**
     * Check if this box is valid.
     *
     * Valid box has: upper.c0 >= lower.c0 && upper.c1 >= lower.c1 && upper.c2 >= lower.c2
     * @return @c true only if this box is valid
     */
    bool isValid() const { return upper.c0 >= lower.c0 && upper.c1 >= lower.c1 && upper.c2 >= lower.c2; }

    /**
     * Check if this box is empty.
     *
     * Empty box has: lower == upper
     * @return @c true only if this box is empty
     */
    bool isEmpty() const { return lower == upper; }

    /**
     * Set this box coordinates to invalid once, so isValid() returns @c false after this call.
     * @see isValid()
     */
    void makeInvalid() { lower = vec(0.0, 0.0, 0.0); upper = vec(-1.0, -1.0, -1.0); }

    /**
     * Calculate area of the box.
     * @return area of the box
     */
    double getArea() const {
        Vec<3, double> v = size();
        return v.c0 * v.c1 * v.c2;
    }

    /**
     * Change i-th coordinate to oposite (mirror).
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * @param flipDir number of coordinate
     */
    inline void flip(size_t flipDir) {
        assert(flipDir < 3);
        double temp = lower[flipDir];
        lower[flipDir] = - upper[flipDir];
        upper[flipDir] = - temp;
    }

    /**
     * Get vector similar to this but with changed i-th coordinate to oposite.
     * WARNING This function does not check if it is valid (for efficiency reasons)
     * @param flipDir number of coordinate
     * @return box similar to this but with mirrored i-th coordinate
     */
    inline Box3D fliped(size_t i) {
        Box3D res = *this;
        res.flip(i);
        return res;
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
    typedef Box2D Box;

    /// Real (double) vector type in 2d space.
    typedef Vec<2,double> DVec;

    /// Number of dimensions (2).
    static const int dim = 2;

    /// Zeroed 2d vector.
    static const DVec ZERO_VEC;

    /// NaNed 2d vector.
    static const DVec NAN_VEC;

    enum Direction {
        DIRECTION_TRAN = 0,
        DIRECTION_VERT = 1
    };

    static void ensureIsValidDirection(unsigned direction) {
        if (direction > 1) throw Exception("Bad 2D direction index, %1% was given but allowed are: 0, 1.", direction);
    }
};

/**
 * Specialization of Primitive, which define types of primitives and constants in space with 3 dimensions.
 */
template <>
struct Primitive<3> {

    /// Rectangle type (cuboid) in 3d space.
    typedef Box3D Box;

    /// Real (double) vector type in 3d space.
    typedef Vec<3,double> DVec;

    /// Number of dimensions (3).
    static const int dim = 3;

    /// Zeroed 3d vector.
    static const DVec ZERO_VEC;

    /// NaNed 3d vector.
    static const DVec NAN_VEC;

    enum Direction {
        DIRECTION_LONG = 0,
        DIRECTION_TRAN = 1,
        DIRECTION_VERT = 2
    };

    static void ensureIsValidDirection(unsigned direction) {
        if (direction > 2)
            throw DimensionError("Bad 3D direction index, %s was given but allowed are: 0, 1, 2.", direction);
    }

    static void ensureIsValid2DDirection(unsigned direction) {
        if (direction != DIRECTION_TRAN && direction != DIRECTION_VERT)
            throw DimensionError("bad 2D direction index, %s was given but allowed are: 1 (DIRECTION_TRAN), 2 (DIRECTION_VERT).", direction);
    }
};

constexpr inline Primitive<3>::Direction direction3D(Primitive<2>::Direction dir2D) {
    return Primitive<3>::Direction(dir2D + 1);
}

constexpr inline Primitive<3>::Direction direction3D(Primitive<3>::Direction dir3D) {
    return dir3D;
}

template <int dim, typename Primitive<dim>::Direction dirToSkip>
struct DirectionWithout {};

template <>
struct DirectionWithout<2, Primitive<2>::DIRECTION_TRAN> {
    static const Primitive<2>::Direction value = Primitive<2>::DIRECTION_VERT;
    static const Primitive<3>::Direction value3d = Primitive<3>::DIRECTION_VERT;
};

template <>
struct DirectionWithout<2, Primitive<2>::DIRECTION_VERT> {
    static const Primitive<2>::Direction value = Primitive<2>::DIRECTION_TRAN;
    static const Primitive<3>::Direction value3d = Primitive<3>::DIRECTION_TRAN;
};

template <>
struct DirectionWithout<3, Primitive<3>::DIRECTION_LONG> {
    static const unsigned value = Primitive<3>::DIRECTION_TRAN | Primitive<3>::DIRECTION_VERT;
    static const Primitive<3>::Direction valueLower = Primitive<3>::DIRECTION_TRAN;
    static const Primitive<3>::Direction valueHigher = Primitive<3>::DIRECTION_VERT;

    //should be not usedm but sometimes it is usefull to make compilation possible
    static const Primitive<2>::Direction value2D = Primitive<2>::Direction(Primitive<2>::DIRECTION_VERT | Primitive<2>::DIRECTION_VERT);
};

template <>
struct DirectionWithout<3, Primitive<3>::DIRECTION_TRAN> {
    static const unsigned value = Primitive<3>::DIRECTION_LONG | Primitive<3>::DIRECTION_VERT;
    static const Primitive<3>::Direction valueLower = Primitive<3>::DIRECTION_LONG;
    static const Primitive<3>::Direction valueHigher = Primitive<3>::DIRECTION_VERT;

    static const Primitive<2>::Direction value2D = Primitive<2>::DIRECTION_VERT;
};

template <>
struct DirectionWithout<3, Primitive<3>::DIRECTION_VERT> {
    static const unsigned value = Primitive<3>::DIRECTION_LONG | Primitive<3>::DIRECTION_TRAN;
    static const Primitive<3>::Direction valueLower = Primitive<3>::DIRECTION_LONG;
    static const Primitive<3>::Direction valueHigher = Primitive<3>::DIRECTION_TRAN;

    static const Primitive<2>::Direction value2D = Primitive<2>::DIRECTION_TRAN;
};


} // namespace plask

#endif

