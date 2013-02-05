#ifndef PLASK__GEOMETRY_ALIGN_H
#define PLASK__GEOMETRY_ALIGN_H

/** @file
This file includes aligners.
*/

#include "transform.h"
#include <boost/lexical_cast.hpp>
#include "../utils/xml.h"
#include "../memory.h"
#include "../utils/metaprog.h"

namespace plask {

namespace align {

/**
 * Directions of aligners activity, same as vec<3, T> directions.
 */
typedef Primitive<3>::Direction Direction;

/// Convert Direction to 2D vector direction
template <Direction direction>
struct DirectionTo2D {
    //static_assert(false, "given 3D direction cannot be converted to 2D direction");
};

template <>
struct DirectionTo2D<Primitive<3>::DIRECTION_TRAN> {
    enum { value = 0 };
};

template <>
struct DirectionTo2D<Primitive<3>::DIRECTION_VERT> {
    enum { value = 1 };
};


/**
 * Helper which allow to implement base class for aligners which work in one direction.
 * Don't use it directly, use Aligner instead.
 * @tparam _direction direction of activity
 * @see Aligner
 */
template <Direction _direction>
struct AlignerImpl: public Printable {

    /// Direction of activity.
    static const Direction direction = _direction;

    /// Coordinate to which this aligner align.
    double coordinate;

    /**
     * Construct new aligner.
     * @param coordinate coordinate to which this aligner align.
     */
    AlignerImpl(double coordinate): coordinate(coordinate) {}

    virtual ~AlignerImpl() {}

    /**
     * Get translation for aligned obiect.
     * @param low, hi aligned object bounds in direction of activity of this aligner
     * @return aligned obiect translation in direction of activity
     */
    virtual double getAlign(double low, double hi) const = 0;

    /**
     * Check if this aligner getAlign use bounds (low and hi parameters) in calculation.
     * @return @c true only if this aligner use bounds, @c false if bounds are ignored
     */
    virtual bool useBounds() const { return true; }

    /**
     * Set object coordinate in direction of aligner activity.
     *
     * This version is called if caller knows child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     * @param childBoundingBox bounding box of object to align
     */
    inline double align(Translation<3>& toAlign, const Box3D& childBoundingBox) const {
        return toAlign.translation[direction] = this->getAlign(childBoundingBox.lower[direction], childBoundingBox.upper[direction]);
    }

    /**
     * Set object translation in direction of aligner activity.
     *
     * This version is called if caller doesn't know child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     */
    virtual double align(Translation<3>& toAlign) const {
        if (this->useBounds())
            return align(toAlign, toAlign.getChild()->getBoundingBox());
        else
            return toAlign.translation[direction] = this->getAlign(0.0, 0.0);
    }

    /**
     * Get aligner name
     * \param axis_names name of axes
     * \return name of the aligner
     */
    virtual std::string key(const AxisNames& axis_names) const = 0;

    /**
     * Get aligner as dictionary
     * \param axis_names name of axes
     * \return string:double map representing the aligner
     */
    std::map<std::string,double> asDict(const AxisNames& axis_names) const {
        std::map<std::string,double> dict;
        dict[key(axis_names)] = this->coordinate;
        return dict;
    }

    /**
     * Write this aligner to XML.
     * @param dest tag where attributes describing this should be appended
     * @param axis_names name of axes
     */
    void writeToXML(XMLElement& dest, const AxisNames& axis_names) const {
        dest.attr(key(axis_names), this->coordinate);
    }

};

template <Direction _direction>
struct AlignerBase: public HolderRef<AlignerImpl<_direction>> {

    AlignerBase() {}

    AlignerBase(AlignerImpl<_direction>* impl): HolderRef<AlignerImpl<_direction>>(impl) {}

    /// Direction of activity.
    static const Direction direction = _direction;

    /**
     * Get coordinate to which this aligner align.
     * @return coordinate to which this aligner align.
     */
    double getCoordinate() const { return this->held->coordinate; }

    /**
     * Get translation for aligned obiect.
     * @param low, hi aligned object bounds in direction of activity of this aligner
     * @return aligned obiect translation in direction of activity
     */
    double getAlign(double low, double hi) const { return this->held->getAlign(low, hi); }

    /**
     * Check if this aligner getAlign use bounds (low and hi parameters) in calculation.
     * @return @c true only if this aligner use bounds, @c false if bounds are ignored
     */
    bool useBounds() const { return this->held->useBounds(); }

    /**
     * Set object coordinate in direction of aligner activity.
     *
     * This version is called if caller knows child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     * @param childBoundingBox bounding box of object to align
     */
    double align(Translation<3>& toAlign, const Box3D& childBoundingBox) const {
        return this->held->align(toAlign, childBoundingBox);
    }

    /**
     * Set object translation in direction of aligner activity.
     *
     * This version is called if caller doesn't know child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     */
    virtual double align(Translation<3>& toAlign) const {
        return this->held->align(toAlign);
    }

    /**
     * Get aligner name
     * \param axis_names name of axes
     * \return name of the aligner
     */
    virtual std::string key(const AxisNames& axis_names) const { return this->held->key(axis_names); }

    /**
     * Get aligner as dictionary
     * \param axis_names name of axes
     * \return string:double map representing the aligner
     */
    std::map<std::string,double> asDict(const AxisNames& axis_names) const {
        return this->held->asDict(axis_names);
    }


    /**
     * Write this aligner to XML.
     * @param dest tag where attributes describing this should be appended
     * @param axis_names name of axes
     */
    void writeToXML(XMLElement& dest, const AxisNames& axis_names) const {
        this->held->writeToXML(dest, axis_names);
    }

    /**
     * Print this to stream.
     * @param out print destination, output stream
     * @param to_print aligner to print
     * @return out stream
     */
    friend inline std::ostream& operator<<(std::ostream& out, const AlignerBase<_direction>& to_print) {
        return out << *to_print.held;
    }

    /**
     * Get string representation of this using print method.
     * @return string representation of this
     */
    std::string str() const { return this->held; }

};

template <Direction... directions> struct Aligner;

/**
 * Base class for one direction aligners (in 2D and 3D spaces).
 */
template <Direction _direction>
struct Aligner<_direction>: public AlignerBase<_direction> {

    static const Direction direction = _direction;

    enum { direction2D = DirectionTo2D<_direction>::value };

    Aligner<_direction>() {}

    Aligner<_direction>(AlignerImpl<_direction>* impl): AlignerBase<_direction>(impl) {}

    using AlignerBase<_direction>::align;

    /**
     * Set object coordinate in direction of aligner activity.
     *
     * This version is called if caller knows child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     * @param childBoundingBox bounding box of object to align
     */
    void align(Translation<2>& toAlign, const Box2D& childBoundingBox) const {
        toAlign.translation[direction2D] = this->getAlign(childBoundingBox.lower[direction2D], childBoundingBox.upper[direction2D]);
    }

    /**
     * Set object translation in direction of aligner activity.
     *
     * This version is called if caller doesn't know child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     */
    void align(Translation<2>& toAlign) const {
        if (this->useBounds())
            this->align(toAlign, toAlign.getChild()->getBoundingBox());
        else
            toAlign.translation[direction2D] = this->getAlign(0.0, 0.0);
    }

};

template <>
struct Aligner<Primitive<3>::DIRECTION_LONG>: public AlignerBase<Primitive<3>::DIRECTION_LONG> {
    Aligner() {}
    Aligner(AlignerImpl<direction>* impl): AlignerBase<Primitive<3>::DIRECTION_LONG>(impl) {}
};

namespace details {

/**
 * Alginer which place zero of object in constant, chosen place.
 */
template <Direction direction>
struct PositionAlignerImpl: public AlignerImpl<direction> {

    PositionAlignerImpl(double translation): AlignerImpl<direction>(translation) {}

    virtual double getAlign(double low, double hi) const {
        return this->coordinate;
    }

    bool useBounds() const { return false; }

    virtual void print(std::ostream& out) const { out << "align object position along axis " << direction << " to " << this->coordinate; }

    virtual std::string key(const AxisNames& axis_names) const { return axis_names[direction]; }
};

}   // namespace details

/**
 * Base class for two directions aligner in 3D space, compose and use two 2D aligners.
 */
template <Direction _direction1, Direction _direction2>
struct AlignerBase2D {

    static_assert(_direction1 < _direction2, "Wrong Aligner template parameters, two different directions are required and first direction must have lower index than second one. Try swapping Aligner template parameters.");

    Aligner<_direction1> dir1aligner;
    Aligner<_direction2> dir2aligner;

    static const Direction direction1 = _direction1, direction2 = _direction2;

    AlignerBase2D() {}

    AlignerBase2D(const Aligner<direction1>& dir1aligner, const Aligner<direction2>& dir2aligner)
        : dir1aligner(dir1aligner), dir2aligner(dir2aligner) {}

    /* Aligner(const Aligner<direction1, direction2>& toCopy)
        : dir1aligner(toCopy.dir1aligner->clone()), dir2aligner(toCopy.dir2aligner->clone()) {}

    Aligner(const Aligner<direction2, direction1>& toCopy)
        : dir1aligner(toCopy.dir2aligner->clone()), dir2aligner(toCopy.dir1aligner->clone()) {}

    Aligner(Aligner<direction1, direction2>&& toMove)
        : dir1aligner(toMove.dir1aligner), dir2aligner(toMove.dir2aligner) {
        toMove.dir1aligner = 0; toMove.dir2aligner = 0;
    }

    Aligner(Aligner<direction2, direction1>&& toMove)
        : dir1aligner(toMove.dir2aligner), dir2aligner(toMove.dir1aligner) {
        toMove.dir1aligner = 0; toMove.dir2aligner = 0;
    }*/

    /**
     * Check if this aligner getAlign use bounding box in calculation.
     * @return @c true only if this aligner use bounding box, @c false if is ignored
     */
    bool useBounds() const { return dir1aligner.useBounds() || dir2aligner.useBounds(); }

    /**
     * Set object translation in directions of aligner activity.
     *
     * This version is called if caller knows child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     * @param childBoundingBox bounding box of object to align
     */
    virtual void align(Translation<3>& toAlign, const Box3D& childBoundingBox) const {
         toAlign.translation[direction1] =
                 dir1aligner.getAlign(childBoundingBox.lower[direction1], childBoundingBox.upper[direction1]);
         toAlign.translation[direction2] =
                 dir2aligner.getAlign(childBoundingBox.lower[direction2], childBoundingBox.upper[direction2]);
    }

    /**
     * Set object translation in directions of aligner activity.
     *
     * This version is called if caller doesn't know child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     */
    virtual void align(Translation<3>& toAlign) const {
        if (useBounds())
            align(toAlign, toAlign.getChild()->getBoundingBox());
        else {
            toAlign.translation[direction1] = dir1aligner.getAlign(0.0, 0.0);
            toAlign.translation[direction2] = dir2aligner.getAlign(0.0, 0.0);
        }
    }

   /**
    * Print this to stream.
    * @param out print destination, output stream
    * @param to_print aligner to print
    * @return out stream
    */
    friend inline std::ostream& operator<<(std::ostream& out, const Aligner<_direction1,_direction2>& to_print) {
        return out << to_print.dir1aligner << ", " << to_print.dir2aligner;
    }

   /**
    * Get string representation of this using print method.
    * @return string representation of this
    */
    std::string str() const { return boost::lexical_cast<std::string>(*this); }

    /**
     * Get aligner as dictionary
     * \param axis_names name of axes
     * \return string:double map representing the aligner
     */
    virtual std::map<std::string,double> asDict(const AxisNames& axis_names) const {
        std::map<std::string,double> dict;
        dict[dir1aligner.key(axis_names)] = dir1aligner.getCoordinate();
        dict[dir2aligner.key(axis_names)] = dir2aligner.getCoordinate();
        return dict;
    }

    /**
     * Write this aligner to XML.
     * @param dest tag where attributes describing this should be appended
     * @param axis_names name of axes
     */
    virtual void writeToXML(XMLElement& dest, const AxisNames& axis_names) const {
        dir1aligner.writeToXML(dest, axis_names);
        dir2aligner.writeToXML(dest, axis_names);
    }

    bool isNull() { return dir1aligner.isNull() || dir2aligner.isNull(); }

};

/**
 * Two directions aligner in 3D space, compose and use two 1D aligners.
 */
template <Direction _direction1, Direction _direction2>
struct Aligner<_direction1, _direction2>: public AlignerBase2D<_direction1, _direction2> {

    Aligner() {}

    Aligner(const Aligner<_direction1>& dir1aligner, const Aligner<_direction2>& dir2aligner)
        : AlignerBase2D<_direction1, _direction2>(dir1aligner, dir2aligner) {}

};

/**
 * Two directions aligner in 2D and 3D space, compose and use two 1D aligners.
 *
 * Version in direction tran-vert can align in 2D translations.
 */
template <>
struct Aligner<Primitive<3>::DIRECTION_TRAN, Primitive<3>::DIRECTION_VERT>: public AlignerBase2D<Primitive<3>::DIRECTION_TRAN, Primitive<3>::DIRECTION_VERT> {

    Aligner() {}

    Aligner(const Aligner<Primitive<3>::DIRECTION_TRAN>& dir1aligner, const Aligner<Primitive<3>::DIRECTION_VERT>& dir2aligner)
        : AlignerBase2D<Primitive<3>::DIRECTION_TRAN, Primitive<3>::DIRECTION_VERT>(dir1aligner, dir2aligner) {}

    using AlignerBase2D<Primitive<3>::DIRECTION_TRAN, Primitive<3>::DIRECTION_VERT>::align;

    /**
     * Set object coordinate in direction of aligner activity.
     *
     * This version is called if caller knows child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     * @param childBoundingBox bounding box of object to align
     */
    void align(Translation<2>& toAlign, const Box2D& childBoundingBox) const {
        toAlign.translation[0] = this->dir1aligner.getAlign(childBoundingBox.lower[0], childBoundingBox.upper[0]);
        toAlign.translation[1] = this->dir2aligner.getAlign(childBoundingBox.lower[1], childBoundingBox.upper[1]);
    }

    /**
     * Set object translation in direction of aligner activity.
     *
     * This version is called if caller doesn't know child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     */
    void align(Translation<2>& toAlign) const {
        if (this->useBounds())
            this->align(toAlign, toAlign.getChild()->getBoundingBox());
        else {
            toAlign.translation[0] = this->dir1aligner.getAlign(0.0, 0.0);
            toAlign.translation[1] = this->dir2aligner.getAlign(0.0, 0.0);
        }
    }

};

/**
 * Three directions aligner in 3D space, compose and use three 1D aligners.
 */
template <>
struct Aligner<> {

    Aligner<Primitive<3>::DIRECTION_LONG> dir1aligner;
    Aligner<Primitive<3>::DIRECTION_TRAN> dir2aligner;
    Aligner<Primitive<3>::DIRECTION_VERT> dir3aligner;

    Aligner() {}

    Aligner(const Aligner<Primitive<3>::DIRECTION_LONG>& dir1aligner, const Aligner<Primitive<3>::DIRECTION_TRAN>& dir2aligner, const Aligner<Primitive<3>::DIRECTION_VERT>& dir3aligner)
        : dir1aligner(dir1aligner), dir2aligner(dir2aligner), dir3aligner(dir3aligner) {}

    /**
     * Check if this aligner getAlign use bounding box in calculation.
     * @return @c true only if this aligner use bounding box, @c false if is ignored
     */
    bool useBounds() const { return dir1aligner.useBounds() || dir2aligner.useBounds() || dir3aligner.useBounds(); }

    /**
     * Set object translation in directions of aligner activity.
     *
     * This version is called if caller knows child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     * @param childBoundingBox bounding box of object to align
     */
    virtual void align(Translation<3>& toAlign, const Box3D& childBoundingBox) const {
         toAlign.translation[0] = dir1aligner.getAlign(childBoundingBox.lower[0], childBoundingBox.upper[0]);
         toAlign.translation[1] = dir2aligner.getAlign(childBoundingBox.lower[1], childBoundingBox.upper[1]);
         toAlign.translation[2] = dir3aligner.getAlign(childBoundingBox.lower[2], childBoundingBox.upper[2]);
    }

    /**
     * Set object translation in directions of aligner activity.
     *
     * This version is called if caller doesn't know child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     */
    virtual void align(Translation<3>& toAlign) const {
        if (useBounds())
            align(toAlign, toAlign.getChild()->getBoundingBox());
        else {
            toAlign.translation[0] = dir1aligner.getAlign(0.0, 0.0);
            toAlign.translation[1] = dir2aligner.getAlign(0.0, 0.0);
            toAlign.translation[2] = dir3aligner.getAlign(0.0, 0.0);
        }
    }

   /**
    * Print this to stream.
    * @param out print destination, output stream
    * @param to_print aligner to print
    * @return out stream
    */
    friend inline std::ostream& operator<<(std::ostream& out, const Aligner<>& to_print) {
        return out << to_print.dir1aligner << ", " << to_print.dir2aligner << ", " << to_print.dir3aligner;
    }

   /**
    * Get string representation of this using print method.
    * @return string representation of this
    */
    std::string str() const { return boost::lexical_cast<std::string>(*this); }

    /**
     * Get aligner as dictionary
     * \param axis_names name of axes
     * \return string:double map representing the aligner
     */
    virtual std::map<std::string,double> asDict(const AxisNames& axis_names) const {
        std::map<std::string,double> dict;
        dict[dir1aligner.key(axis_names)] = dir1aligner.getCoordinate();
        dict[dir2aligner.key(axis_names)] = dir2aligner.getCoordinate();
        dict[dir3aligner.key(axis_names)] = dir3aligner.getCoordinate();
        return dict;
    }

    /**
     * Write this aligner to XML.
     * @param dest tag where attributes describing this should be appended
     * @param axis_names name of axes
     */
    virtual void writeToXML(XMLElement& dest, const AxisNames& axis_names) const {
        dir1aligner.writeToXML(dest, axis_names);
        dir2aligner.writeToXML(dest, axis_names);
        dir3aligner.writeToXML(dest, axis_names);
    }

    bool isNull() { return dir1aligner.isNull() || dir2aligner.isNull() || dir3aligner.isNull(); }

};

//two 1D aligners gives 2D aligner
inline Aligner<Primitive<3>::Direction(0), Primitive<3>::Direction(1)> operator&(const Aligner<Primitive<3>::Direction(0)>& dir1aligner, const Aligner<Primitive<3>::Direction(1)>& dir2aligner) {
    return Aligner<Primitive<3>::Direction(0), Primitive<3>::Direction(1)>(dir1aligner, dir2aligner);
}

inline Aligner<Primitive<3>::Direction(0), Primitive<3>::Direction(1)> operator&(const Aligner<Primitive<3>::Direction(1)>& dir1aligner, const Aligner<Primitive<3>::Direction(0)>& dir2aligner) {
    return Aligner<Primitive<3>::Direction(0), Primitive<3>::Direction(1)>(dir2aligner, dir1aligner);
}

inline Aligner<Primitive<3>::Direction(0), Primitive<3>::Direction(2)> operator&(const Aligner<Primitive<3>::Direction(0)>& dir1aligner, const Aligner<Primitive<3>::Direction(2)>& dir2aligner) {
    return Aligner<Primitive<3>::Direction(0), Primitive<3>::Direction(2)>(dir1aligner, dir2aligner);
}

inline Aligner<Primitive<3>::Direction(0), Primitive<3>::Direction(2)> operator&(const Aligner<Primitive<3>::Direction(2)>& dir1aligner, const Aligner<Primitive<3>::Direction(0)>& dir2aligner) {
    return Aligner<Primitive<3>::Direction(0), Primitive<3>::Direction(2)>(dir2aligner, dir1aligner);
}

inline Aligner<Primitive<3>::Direction(1), Primitive<3>::Direction(2)> operator&(const Aligner<Primitive<3>::Direction(1)>& dir1aligner, const Aligner<Primitive<3>::Direction(2)>& dir2aligner) {
    return Aligner<Primitive<3>::Direction(1), Primitive<3>::Direction(2)>(dir1aligner, dir2aligner);
}

inline Aligner<Primitive<3>::Direction(1), Primitive<3>::Direction(2)> operator&(const Aligner<Primitive<3>::Direction(2)>& dir1aligner, const Aligner<Primitive<3>::Direction(1)>& dir2aligner) {
    return Aligner<Primitive<3>::Direction(1), Primitive<3>::Direction(2)>(dir2aligner, dir1aligner);
}

//1D aligner & 2D aligner = 3D aligner
inline Aligner<> operator&(const Aligner<Primitive<3>::Direction(1), Primitive<3>::Direction(2)>& dir12aligner, const Aligner<Primitive<3>::Direction(0)>& dir0aligner) {
    return Aligner<>(dir0aligner, dir12aligner.dir1aligner, dir12aligner.dir2aligner);
}

inline Aligner<> operator&(const Aligner<Primitive<3>::Direction(0)>& dir0aligner, const Aligner<Primitive<3>::Direction(1), Primitive<3>::Direction(2)>& dir12aligner) {
    return dir12aligner & dir0aligner;
}

inline Aligner<> operator&(const Aligner<Primitive<3>::Direction(0), Primitive<3>::Direction(2)>& dir02aligner, const Aligner<Primitive<3>::Direction(1)>& dir1aligner) {
    return Aligner<>(dir02aligner.dir1aligner, dir1aligner, dir02aligner.dir2aligner);
}

inline Aligner<> operator&(const Aligner<Primitive<3>::Direction(1)>& dir1aligner, const Aligner<Primitive<3>::Direction(0), Primitive<3>::Direction(2)>& dir02aligner) {
    return dir02aligner & dir1aligner;
}

inline Aligner<> operator&(const Aligner<Primitive<3>::Direction(0), Primitive<3>::Direction(1)>& dir01aligner, const Aligner<Primitive<3>::Direction(2)>& dir2aligner) {
    return Aligner<>(dir01aligner.dir1aligner, dir01aligner.dir2aligner, dir2aligner);
}

inline Aligner<> operator&(const Aligner<Primitive<3>::Direction(2)>& dir2aligner, const Aligner<Primitive<3>::Direction(0), Primitive<3>::Direction(1)>& dir01aligner) {
    return dir01aligner & dir2aligner;
}


namespace details {

typedef double alignStrategy(double lo, double hi, double coordinate);
inline double lowToCoordinate(double lo, double hi, double coordinate) { return coordinate -lo; }
inline double hiToCoordinate(double lo, double hi, double coordinate) { return coordinate -hi; }
inline double centerToCoordinate(double lo, double hi, double coordinate) { return coordinate -(lo+hi)/2.0; }

struct LEFT { static constexpr const char* value = "left"; };
struct RIGHT { static constexpr const char* value = "right"; };
struct FRONT { static constexpr const char* value = "front"; };
struct BACK { static constexpr const char* value = "back"; };
struct TOP { static constexpr const char* value = "top"; };
struct BOTTOM { static constexpr const char* value = "bottom"; };
struct TRAN_CENTER { static constexpr const char* value = "trancenter"; };
struct LON_CENTER { static constexpr const char* value = "longcenter"; };
struct VERT_CENTER { static constexpr const char* value = "vertcenter"; };

template <Direction direction, alignStrategy strategy, typename name_tag>
struct AlignerCustomImpl: public AlignerImpl<direction> {

    AlignerCustomImpl(double coordinate): AlignerImpl<direction>(coordinate) {}

    virtual double getAlign(double low, double hi) const {
        return strategy(low, hi, this->coordinate);
    }

    /*virtual AlignerImpl<direction, strategy, name_tag>* clone() const {
        return new AlignerImpl<direction, strategy, name_tag>(this->coordinate);
    }*/

    virtual void print(std::ostream& out) const { out << "align " << name_tag::value << " to " << this->coordinate; }

    virtual std::string key(const AxisNames& axis_names) const { return name_tag::value; }
};

}   // namespace details

// 1D tran. aligners:
inline Aligner<Primitive<3>::DIRECTION_TRAN> left(double coordinate) { return new details::AlignerCustomImpl<Primitive<3>::DIRECTION_TRAN, details::lowToCoordinate, details::LEFT>(coordinate); }
inline Aligner<Primitive<3>::DIRECTION_TRAN> right(double coordinate) { return new details::AlignerCustomImpl<Primitive<3>::DIRECTION_TRAN, details::hiToCoordinate, details::RIGHT>(coordinate); }
inline Aligner<Primitive<3>::DIRECTION_TRAN> tranCenter(double coordinate) { return new details::AlignerCustomImpl<Primitive<3>::DIRECTION_TRAN, details::centerToCoordinate, details::TRAN_CENTER>(coordinate); }
inline Aligner<Primitive<3>::DIRECTION_TRAN> center(double coordinate) { return new details::AlignerCustomImpl<Primitive<3>::DIRECTION_TRAN, details::centerToCoordinate, details::TRAN_CENTER>(coordinate); }
inline Aligner<Primitive<3>::DIRECTION_TRAN> tran(double coordinate) { return new details::PositionAlignerImpl<Primitive<3>::DIRECTION_TRAN>(coordinate); }

// 1D long. aligners:
inline Aligner<Primitive<3>::DIRECTION_LONG> front(double coordinate) { return new details::AlignerCustomImpl<Primitive<3>::DIRECTION_LONG, details::hiToCoordinate, details::FRONT>(coordinate); }
inline Aligner<Primitive<3>::DIRECTION_LONG> back(double coordinate) { return new details::AlignerCustomImpl<Primitive<3>::DIRECTION_LONG, details::lowToCoordinate, details::BACK>(coordinate); }
inline Aligner<Primitive<3>::DIRECTION_LONG> lonCenter(double coordinate) { return new details::AlignerCustomImpl<Primitive<3>::DIRECTION_LONG, details::centerToCoordinate, details::LON_CENTER>(coordinate); }
inline Aligner<Primitive<3>::DIRECTION_LONG> lon(double coordinate) { return new details::PositionAlignerImpl<Primitive<3>::DIRECTION_LONG>(coordinate); }

// 1D vert. aligners:
inline Aligner<Primitive<3>::DIRECTION_VERT> bottom(double coordinate) { return new details::AlignerCustomImpl<Primitive<3>::DIRECTION_VERT, details::lowToCoordinate, details::BOTTOM>(coordinate); }
inline Aligner<Primitive<3>::DIRECTION_VERT> top(double coordinate) { return new details::AlignerCustomImpl<Primitive<3>::DIRECTION_VERT, details::hiToCoordinate, details::TOP>(coordinate); }
inline Aligner<Primitive<3>::DIRECTION_VERT> vertCenter(double coordinate) { return new details::AlignerCustomImpl<Primitive<3>::DIRECTION_VERT, details::centerToCoordinate, details::VERT_CENTER>(coordinate); }
inline Aligner<Primitive<3>::DIRECTION_VERT> vert(double coordinate) { return new details::PositionAlignerImpl<Primitive<3>::DIRECTION_VERT>(coordinate); }

inline Aligner<Primitive<3>::DIRECTION_TRAN, Primitive<3>::DIRECTION_VERT> fromVector(const Vec<2, double>& v) {
    return tran(v.tran()) & vert(v.vert());
}

inline Aligner<> fromVector(const Vec<3, double>& v) {
    return lon(v.lon()) & tran(v.tran()) & vert(v.vert());
}

template<Primitive<3>::Direction dir>
using LowerBoundImpl = details::AlignerCustomImpl<dir, details::lowToCoordinate, typename chooseType<dir, details::BACK, details::LEFT, details::BOTTOM>::type>;

/**
 * Get aligner with align in one direction to lower bound.
 * @return aligner with align to lower bound in given direction, equal to left, back, or bottom.
 * @tparam dir direction
 */
template <Primitive<3>::Direction dir> inline Aligner<dir> lowerBoundZero() { return new LowerBoundImpl<dir>(0.0); }
template <Primitive<3>::Direction dir1, Primitive<3>::Direction dir2> inline Aligner<dir1, dir2> lowerBoundZero() { return lowerBoundZero<dir1> & lowerBoundZero<dir2>; }

typedef std::function<boost::optional<double>(const std::string& name)> Dictionary;

namespace details {
    Aligner<Primitive<3>::DIRECTION_TRAN> transAlignerFromDictionary(Dictionary dic, const std::string& axis_name);
    Aligner<Primitive<3>::DIRECTION_LONG> lonAlignerFromDictionary(Dictionary dic, const std::string& axis_name);
    Aligner<Primitive<3>::DIRECTION_VERT> vertAlignerFromDictionary(Dictionary dic, const std::string& axis_name);
}

/**
 * Construct aligner in given direction from dictionary.
 *
 * Throw excpetion if @p dic includes information about multiple aligners in given @p direction.
 * @param dictionary dictionary which can describe aligner
 * @param axis_name name of axis in given @p direction
 * @return parsed aligner or @c nullptr if no information about aligner was found in @p dictionary
 * @tparam direction direction
 */
template <Direction direction>
Aligner<direction> fromDictionary(Dictionary dictionary, const std::string& axis_name);

template <>
inline Aligner<Primitive<3>::DIRECTION_TRAN> fromDictionary<Primitive<3>::DIRECTION_TRAN>(Dictionary dictionary, const std::string& axis_name) {
    return details::transAlignerFromDictionary(dictionary, axis_name);
}

template <>
inline Aligner<Primitive<3>::DIRECTION_LONG> fromDictionary<Primitive<3>::DIRECTION_LONG>(Dictionary dictionary, const std::string& axis_name) {
    return details::lonAlignerFromDictionary(dictionary, axis_name);
}

template <>
inline Aligner<Primitive<3>::DIRECTION_VERT> fromDictionary<Primitive<3>::DIRECTION_VERT>(Dictionary dictionary, const std::string& axis_name) {
    return details::vertAlignerFromDictionary(dictionary, axis_name);
}

/**
 * Construct aligner in given direction from dictionary.
 *
 * Throw exception if @p dic includes information about multiple aligners in given @p direction.
 * @param dictionary dictionary which can describe aligner
 * @param axis_names names of axes
 * @return parsed aligner or @c nullptr if no information about aligner was found in @p dictionary
 * @tparam direction direction
 */
template <Direction direction>
Aligner<direction> fromDictionary(Dictionary dic, const AxisNames& axis_names) {
    return fromDictionary<direction>(dic, axis_names[direction]);
}

/**
 * Construct aligner in given direction from dictionary.
 *
 * Throw exception if @p dic includes information about multiple aligners in given @p direction.
 * @param dictionary dictionary which can describes 2D aligner
 * @param axis_names names of axes
 * @param default_aligner default aligners, returned when there is no information about aligner in @p dictionary
 * @return parsed aligner or @p default_aligner if no information about aligner was found in @p dictionary
 * @tparam direction direction
 */
template <Direction direction>
Aligner<direction> fromDictionary(Dictionary dictionary, const AxisNames& axis_names, Aligner<direction> default_aligner) {
    Aligner<direction> result = fromDictionary<direction>(dictionary, axis_names[direction]);
    if (result.isNull()) result = default_aligner;
    return result;
}

/**
 * Allow to use XMLReader as dictionary which is required by aligners parsing functions.
 */
struct DictionaryFromXML {
    const XMLReader& reader;
    DictionaryFromXML(const XMLReader& reader): reader(reader) {}
    boost::optional<double> operator()(const std::string& s) const { return reader.getAttribute<double>(s); }
};

template <Direction direction>
inline Aligner<direction> fromXML(const XMLReader& reader, const std::string& axis_name) {
    return fromDictionary<direction>(DictionaryFromXML(reader), axis_name);
}

template <Direction direction>
inline Aligner<direction> fromXML(const XMLReader& reader, const AxisNames& axis_names) {
    return fromXML<direction>(reader, axis_names[direction]);
}

template <Direction direction>
inline Aligner<direction> fromXML(const XMLReader& reader, const AxisNames& axis_names, Aligner<direction> default_aligner) {
    return fromDictionary<direction>(DictionaryFromXML(reader), axis_names, default_aligner);
}

/**
 * Construct 2 directions aligner from dictionary.
 *
 * Throw exception if there is no information about one of this direction in @p dic,
 * or when some direction is described multiple times.
 * @param dic dictionary which describes required aligner in 2 directions
 * @param axes names of axes
 * @return constructed aligner
 */
template <Direction direction1, Direction direction2>
inline Aligner<direction1, direction2> fromDictionary(Dictionary dic, const AxisNames& axes) {
    Aligner<direction1> a1 = fromDictionary<direction1>(dic, axes);
    if (a1.isNull()) throw Exception("No aligner for axis%1% defined.", direction1);
    Aligner<direction2> a2 = fromDictionary<direction2>(dic, axes);
    if (a2.isNull()) throw Exception("No aligner for axis%1% defined.", direction2);
    return Aligner<direction1, direction2>(a1, a2);
}

/**
 * Construct 2 directions aligner from XML.
 *
 * Throw exception if there is no information about one of this direction in @p reader,
 * or when some direction is described multiple times.
 * @param reader XML reader which point to tag which attributes describe required aligner in 2 directions
 * @param axes names of axes
 * @return constructed aligner
 */
template <Direction direction1, Direction direction2>
inline Aligner<direction1, direction2> fromXML(const XMLReader& reader, const AxisNames& axes) {
    return fromDictionary<direction1, direction2>(DictionaryFromXML(reader), axes);
}

/**
 * Construct 2 direction aligner from dictionary.
 *
 * Throw exception if some direction is described multiple times.
 * @param dictionary dictionary which describes required aligner in 2 directions
 * @param axes names of axes
 * @param default2D default aligners, used when there is no information about aligner in some direction in @p dictionary
 * @return constructed aligner
 */
template <Direction direction1, Direction direction2>
inline Aligner<direction1, direction2> fromDictionary(Dictionary dictionary, const AxisNames& axes, Aligner<direction1, direction2> default2D) {
    return Aligner<direction1, direction2>(fromDictionary(dictionary, axes, default2D.dir1aligner), fromDictionary(dictionary, axes, default2D.dir2aligner));
}

/**
 * Construct 2 direction aligner from XML.
 *
 * Throw exception if some direction is described multiple times.
 * @param reader XML reader which point to tag which attributes describe required aligner in 2 directions
 * @param axes names of axes
 * @param default2D default aligners, used when there is no information about aligner in some direction in @p reader
 * @return constructed aligner
 */
template <Direction direction1, Direction direction2>
inline Aligner<direction1, direction2> fromXML(const XMLReader& reader, const AxisNames& axes, Aligner<direction1, direction2> default2D) {
    return fromDictionary<direction1, direction2>(DictionaryFromXML(reader), axes, default2D);
}

/**
 * Construct 3 direction aligner from dictionary.
 *
 * Throw exception if there is no information about one of this direction in @p dic,
 * or when some direction is described multiple times.
 * @param dictionary dictionary which describes required aligner in 3 directions
 * @param axis_names names of axes
 * @return constructed aligner
 */
inline Aligner<> fromDictionary(Dictionary dictionary, const AxisNames& axis_names) {
    Aligner<Primitive<3>::Direction(0)> a0 = fromDictionary<Primitive<3>::Direction(0)>(dictionary, axis_names);
    if (a0.isNull()) throw Exception("No aligner for axis%1% defined.", 0);
    Aligner<Primitive<3>::Direction(1)> a1 = fromDictionary<Primitive<3>::Direction(1)>(dictionary, axis_names);
    if (a1.isNull()) throw Exception("No aligner for axis%1% defined.", 1);
    Aligner<Primitive<3>::Direction(2)> a2 = fromDictionary<Primitive<3>::Direction(2)>(dictionary, axis_names);
    if (a2.isNull()) throw Exception("No aligner for axis%1% defined.", 2);
    return Aligner<>(a0, a1, a2);
}

/**
 * Construct 3 direction aligner from XML.
 *
 * Throw exception if there is no information about one of this direction in @p reader,
 * or when some direction is described multiple times.
 * @param reader XML reader which point to tag which attributes describe required aligner in 3 directions
 * @param axes names of axes
 * @return constructed aligner
 */
inline Aligner<> fromXML(const XMLReader& reader, const AxisNames& axes) {
    return fromDictionary(DictionaryFromXML(reader), axes);
}

/**
 * Construct 3 direction aligner from dictionary.
 *
 * Throw exception if some direction is described multiple times.
 * @param dic dictionary which describes required aligner in 3 directions
 * @param axes names of axes
 * @param default3D default aligner in all directions, used when there is no information about aligner in some direction in @p dictionary
 * @return constructed aligner
 */
inline Aligner<> fromDictionary(Dictionary dic, const AxisNames& axes, Aligner<> default3D) {
    return Aligner<>(fromDictionary(dic, axes, default3D.dir1aligner), fromDictionary(dic, axes, default3D.dir2aligner), fromDictionary(dic, axes, default3D.dir3aligner));
}

/**
 * Construct 3 direction aligner from XML.
 *
 * Throw exception if some direction is described multiple times.
 * @param reader XML reader which point to tag which attributes describe required aligner in 3 directions
 * @param axes names of axes
 * @param default3D default aligner in all directions, used when there is no information about aligner in some direction in @p reader
 * @return constructed aligner
 */
inline Aligner<> fromXML(const XMLReader& reader, const AxisNames& axes, Aligner<> default3D) {
    return fromDictionary(DictionaryFromXML(reader), axes, default3D);
}

/**
 * Align @p toAlign by aligner1 and aligner2.
 *
 * It is equal to, but possibly faster than:
 * @code
 * aligner1.align(toAlign);
 * aligner2.align(toAlign);
 * @endcode
 * It calculates bounding box of toAlign only once and only if it is required by any aligner.
 *
 * @param toAlign object to align, translation in 2D or 3D
 * @param aligner1, aligner2 aligners
 */
template <typename TranslationT, typename Aligner1T, typename Aligner2T>
void align(TranslationT& toAlign, const Aligner1T& aligner1, const Aligner2T& aligner2) {
    if (aligner1.useBounds() || aligner2.useBounds()) {
        auto bb = toAlign.getChild()->getBoundingBox();
        aligner1.align(toAlign, bb);
        aligner2.align(toAlign, bb);
    } else {
        aligner1.align(toAlign);
        aligner2.align(toAlign);
    }
}

}   // namespace align
}   // namespace plask

#endif // PLASK__GEOMETRY_ALIGN_H
