#ifndef PLASK__GEOMETRY_ALIGN_H
#define PLASK__GEOMETRY_ALIGN_H

/** @file
This file includes aligners.
*/

#include "transform.h"
#include <boost/lexical_cast.hpp>
#include "../utils/xml.h"
#include "../memory.h"

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
 * Don't use it directly, use Aligner1D instead.
 * @tparam _direction direction of activity
 * @see Aligner1D
 */
template <Direction _direction>
struct Aligner1DImpl: public Printable {

    /// Direction of activity.
    static const Direction direction = _direction;

    /// Coordinate to which this aligner align.
    double coordinate;

    /**
     * Construct new aligner.
     * @param coordinate coordinate to which this aligner align.
     */
    Aligner1DImpl(double coordinate): coordinate(coordinate) {}

    virtual ~Aligner1DImpl() {}

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
struct Aligner1DBase: public HolderRef<Aligner1DImpl<_direction>> {

    Aligner1DBase() {}

    Aligner1DBase(Aligner1DImpl<_direction>* impl): HolderRef<Aligner1DImpl<_direction>>(impl) {}

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
    friend inline std::ostream& operator<<(std::ostream& out, const Aligner1DBase<_direction>& to_print) {
        return out << *to_print.held;
    }

    /**
     * Get string representation of this using print method.
     * @return string representation of this
     */
    std::string str() const { return this->held; }

};

template <Direction direction> struct Aligner1D;

/**
 * Base class for one direction aligners (in 2D and 3D spaces).
 */
template <Direction direction>
struct Aligner1D: public Aligner1DBase<direction> {

    enum { direction2D = DirectionTo2D<direction>::value };

    Aligner1D() {}

    Aligner1D(Aligner1DImpl<direction>* impl): Aligner1DBase<direction>(impl) {}

    using Aligner1DBase<direction>::align;

    /**
     * Set object coordinate in direction of aligner activity.
     *
     * This version is called if caller knows child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     * @param childBoundingBox bounding box of object to align
     */
    inline double align(Translation<2>& toAlign, const Box2D& childBoundingBox) const {
        return toAlign.translation[direction2D] = this->getAlign(childBoundingBox.lower[direction2D], childBoundingBox.upper[direction2D]);
    }

    /**
     * Set object translation in direction of aligner activity.
     *
     * This version is called if caller doesn't know child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     */
    double align(Translation<2>& toAlign) const {
        if (this->useBounds())
            return this->align(toAlign, toAlign.getChild()->getBoundingBox());
        else
            return toAlign.translation[direction2D] = this->getAlign(0.0, 0.0);
    }

};

template <>
struct Aligner1D<Primitive<3>::DIRECTION_LONG>: public Aligner1DBase<Primitive<3>::DIRECTION_LONG> {
    Aligner1D() {}
    Aligner1D(Aligner1DImpl<direction>* impl): Aligner1DBase<Primitive<3>::DIRECTION_LONG>(impl) {}
};

namespace details {

/**
 * Alginer which place zero of object in constant, chosen place.
 */
template <Direction direction>
struct PositionAligner1DImpl: public Aligner1DImpl<direction> {

    PositionAligner1DImpl(double translation): Aligner1DImpl<direction>(translation) {}

    virtual double getAlign(double low, double hi) const {
        return this->coordinate;
    }

    bool useBounds() const { return false; }

    virtual void print(std::ostream& out) const { out << "align object position along axis " << direction << " to " << this->coordinate; }

    virtual std::string key(const AxisNames& axis_names) const { return axis_names[direction]; }
};

}   // namespace details

/**
 * Two directions aligner in 3D space, compose and use two 2D aligners.
 */
template <Direction _direction1, Direction _direction2>
class Aligner2D {

    Aligner1D<_direction1> dir1aligner;
    Aligner1D<_direction2> dir2aligner;

public:

    static const Direction direction1 = _direction1, direction2 = _direction2;

    static_assert(_direction1 != _direction2, "Wrong Aligner2D template parameters, two different directions are required.");

   Aligner2D() {}

   Aligner2D(const Aligner1D<direction1>& dir1aligner, const Aligner1D<direction2>& dir2aligner)
        : dir1aligner(dir1aligner), dir2aligner(dir2aligner) {}

    /* Aligner2D(const Aligner2D<direction1, direction2>& toCopy)
        : dir1aligner(toCopy.dir1aligner->clone()), dir2aligner(toCopy.dir2aligner->clone()) {}

    Aligner2D(const Aligner2D<direction2, direction1>& toCopy)
        : dir1aligner(toCopy.dir2aligner->clone()), dir2aligner(toCopy.dir1aligner->clone()) {}

    Aligner2D(Aligner2D<direction1, direction2>&& toMove)
        : dir1aligner(toMove.dir1aligner), dir2aligner(toMove.dir2aligner) {
        toMove.dir1aligner = 0; toMove.dir2aligner = 0;
    }

    Aligner2D(Aligner2D<direction2, direction1>&& toMove)
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
    friend inline std::ostream& operator<<(std::ostream& out, const Aligner2D<_direction1, _direction2>& to_print) {
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

};

inline Aligner2D<Primitive<3>::Direction(0), Primitive<3>::Direction(1)> operator&(const Aligner1D<Primitive<3>::Direction(0)>& dir1aligner, const Aligner1D<Primitive<3>::Direction(1)>& dir2aligner) {
    return Aligner2D<Primitive<3>::Direction(0), Primitive<3>::Direction(1)>(dir1aligner, dir2aligner);
}

inline Aligner2D<Primitive<3>::Direction(0), Primitive<3>::Direction(1)> operator&(const Aligner1D<Primitive<3>::Direction(1)>& dir1aligner, const Aligner1D<Primitive<3>::Direction(0)>& dir2aligner) {
    return Aligner2D<Primitive<3>::Direction(0), Primitive<3>::Direction(1)>(dir2aligner, dir1aligner);
}

inline Aligner2D<Primitive<3>::Direction(0), Primitive<3>::Direction(2)> operator&(const Aligner1D<Primitive<3>::Direction(0)>& dir1aligner, const Aligner1D<Primitive<3>::Direction(2)>& dir2aligner) {
    return Aligner2D<Primitive<3>::Direction(0), Primitive<3>::Direction(2)>(dir1aligner, dir2aligner);
}

inline Aligner2D<Primitive<3>::Direction(0), Primitive<3>::Direction(2)> operator&(const Aligner1D<Primitive<3>::Direction(2)>& dir1aligner, const Aligner1D<Primitive<3>::Direction(0)>& dir2aligner) {
    return Aligner2D<Primitive<3>::Direction(0), Primitive<3>::Direction(2)>(dir2aligner, dir1aligner);
}

inline Aligner2D<Primitive<3>::Direction(1), Primitive<3>::Direction(2)> operator&(const Aligner1D<Primitive<3>::Direction(1)>& dir1aligner, const Aligner1D<Primitive<3>::Direction(2)>& dir2aligner) {
    return Aligner2D<Primitive<3>::Direction(1), Primitive<3>::Direction(2)>(dir1aligner, dir2aligner);
}

inline Aligner2D<Primitive<3>::Direction(1), Primitive<3>::Direction(2)> operator&(const Aligner1D<Primitive<3>::Direction(2)>& dir1aligner, const Aligner1D<Primitive<3>::Direction(1)>& dir2aligner) {
    return Aligner2D<Primitive<3>::Direction(1), Primitive<3>::Direction(2)>(dir2aligner, dir1aligner);
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
struct Aligner1DCustomImpl: public Aligner1DImpl<direction> {

    Aligner1DCustomImpl(double coordinate): Aligner1DImpl<direction>(coordinate) {}

    virtual double getAlign(double low, double hi) const {
        return strategy(low, hi, this->coordinate);
    }

    /*virtual Aligner1DImpl<direction, strategy, name_tag>* clone() const {
        return new Aligner1DImpl<direction, strategy, name_tag>(this->coordinate);
    }*/

    virtual void print(std::ostream& out) const { out << "align " << name_tag::value << " to " << this->coordinate; }

    virtual std::string key(const AxisNames& axis_names) const { return name_tag::value; }
};

}   // namespace details

//2D tran. aligners:
inline Aligner1D<Primitive<3>::DIRECTION_TRAN> left(double coordinate) { return new details::Aligner1DCustomImpl<Primitive<3>::DIRECTION_TRAN, details::lowToCoordinate, details::LEFT>(coordinate); }
inline Aligner1D<Primitive<3>::DIRECTION_TRAN> right(double coordinate) { return new details::Aligner1DCustomImpl<Primitive<3>::DIRECTION_TRAN, details::hiToCoordinate, details::RIGHT>(coordinate); }
inline Aligner1D<Primitive<3>::DIRECTION_TRAN> tranCenter(double coordinate) { return new details::Aligner1DCustomImpl<Primitive<3>::DIRECTION_TRAN, details::centerToCoordinate, details::TRAN_CENTER>(coordinate); }
inline Aligner1D<Primitive<3>::DIRECTION_TRAN> center(double coordinate) { return new details::Aligner1DCustomImpl<Primitive<3>::DIRECTION_TRAN, details::centerToCoordinate, details::TRAN_CENTER>(coordinate); }
inline Aligner1D<Primitive<3>::DIRECTION_TRAN> tran(double coordinate) { return new details::PositionAligner1DImpl<Primitive<3>::DIRECTION_TRAN>(coordinate); }

//2D long. aligners:
inline Aligner1D<Primitive<3>::DIRECTION_LONG> front(double coordinate) { return new details::Aligner1DCustomImpl<Primitive<3>::DIRECTION_LONG, details::hiToCoordinate, details::FRONT>(coordinate); }
inline Aligner1D<Primitive<3>::DIRECTION_LONG> back(double coordinate) { return new details::Aligner1DCustomImpl<Primitive<3>::DIRECTION_LONG, details::lowToCoordinate, details::BACK>(coordinate); }
inline Aligner1D<Primitive<3>::DIRECTION_LONG> lonCenter(double coordinate) { return new details::Aligner1DCustomImpl<Primitive<3>::DIRECTION_LONG, details::centerToCoordinate, details::LON_CENTER>(coordinate); }
inline Aligner1D<Primitive<3>::DIRECTION_LONG> lon(double coordinate) { return new details::PositionAligner1DImpl<Primitive<3>::DIRECTION_LONG>(coordinate); }

//2D vert. aligners:
inline Aligner1D<Primitive<3>::DIRECTION_VERT> bottom(double coordinate) { return new details::Aligner1DCustomImpl<Primitive<3>::DIRECTION_VERT, details::lowToCoordinate, details::BOTTOM>(coordinate); }
inline Aligner1D<Primitive<3>::DIRECTION_VERT> top(double coordinate) { return new details::Aligner1DCustomImpl<Primitive<3>::DIRECTION_VERT, details::hiToCoordinate, details::TOP>(coordinate); }
inline Aligner1D<Primitive<3>::DIRECTION_VERT> vertCenter(double coordinate) { return new details::Aligner1DCustomImpl<Primitive<3>::DIRECTION_VERT, details::centerToCoordinate, details::VERT_CENTER>(coordinate); }
inline Aligner1D<Primitive<3>::DIRECTION_VERT> vert(double coordinate) { return new details::PositionAligner1DImpl<Primitive<3>::DIRECTION_VERT>(coordinate); }

//3D lon/tran aligners:
/*typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_LONG, details::hiToCoordinate, details::FRONT, Primitive<3>::DIRECTION_TRAN, details::lowToCoordinate, details::LEFT> FrontLeft;
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_LONG, details::hiToCoordinate, details::FRONT, Primitive<3>::DIRECTION_TRAN, details::hiToCoordinate, details::RIGHT> FrontRight;
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_LONG, details::hiToCoordinate, details::FRONT, Primitive<3>::DIRECTION_TRAN, details::centerToCoordinate, details::CENTER> FrontCenter;
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_LONG, details::lowToCoordinate, details::BACK, Primitive<3>::DIRECTION_TRAN, details::lowToCoordinate, details::LEFT> BackLeft;
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_LONG, details::lowToCoordinate, details::BACK, Primitive<3>::DIRECTION_TRAN, details::hiToCoordinate, details::RIGHT> BackRight;
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_LONG, details::lowToCoordinate, details::BACK, Primitive<3>::DIRECTION_TRAN, details::centerToCoordinate, details::CENTER> BackCenter;
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_LONG, details::centerToCoordinate, details::CENTER, Primitive<3>::DIRECTION_TRAN, details::lowToCoordinate, details::LEFT> CenterLeft;
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_LONG, details::centerToCoordinate, details::CENTER, Primitive<3>::DIRECTION_TRAN, details::hiToCoordinate, details::RIGHT> CenterRight;
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_LONG, details::centerToCoordinate, details::CENTER, Primitive<3>::DIRECTION_TRAN, details::centerToCoordinate, details::CENTER> CenterCenter;
typedef TranslationAligner2D<Primitive<3>::DIRECTION_LONG, Primitive<3>::DIRECTION_TRAN> LonTran;*/
//typedef Aligner2D<DIR3D_LON, DIR3D_TRAN> NFLR;
//TODO mixed variants

typedef std::function<boost::optional<double>(const std::string& name)> Dictionary;

namespace details {
    Aligner1D<Primitive<3>::DIRECTION_TRAN> transAlignerFromDictionary(Dictionary dic, const std::string& axis_name);
    Aligner1D<Primitive<3>::DIRECTION_LONG> lonAlignerFromDictionary(Dictionary dic, const std::string& axis_name);
    Aligner1D<Primitive<3>::DIRECTION_VERT> vertAlignerFromDictionary(Dictionary dic, const std::string& axis_name);
}

/**
 * Construct 2d aligner in given direction from dictionary.
 *
 * Throw excpetion if @p dic includes information about multiple aligners in given @p direction.
 * @param dictionary dictionary which can describes 2D aligner
 * @param axis_name name of axis in given @p direction
 * @return parsed aligner or nullptr if no information found
 * @tparam direction direction
 */
template <Direction direction>
Aligner1D<direction> fromDictionary(Dictionary dic, const std::string& axis_name);

template <>
inline Aligner1D<Primitive<3>::DIRECTION_TRAN> fromDictionary<Primitive<3>::DIRECTION_TRAN>(Dictionary dic, const std::string& axis_name) {
    return details::transAlignerFromDictionary(dic, axis_name);
}

template <>
inline Aligner1D<Primitive<3>::DIRECTION_LONG> fromDictionary<Primitive<3>::DIRECTION_LONG>(Dictionary dic, const std::string& axis_name) {
    return details::lonAlignerFromDictionary(dic, axis_name);
}

template <>
inline Aligner1D<Primitive<3>::DIRECTION_VERT> fromDictionary<Primitive<3>::DIRECTION_VERT>(Dictionary dic, const std::string& axis_name) {
    return details::vertAlignerFromDictionary(dic, axis_name);
}

/**
 * Construct 2d aligner in given direction from dictionary.
 *
 * Throw exception if @p dic includes information about multiple aligners in given @p direction.
 * @param dictionary dictionary which can describes 2D aligner
 * @param axis_names names of axes
 * @return parsed aligner or nullptr if no information found
 * @tparam direction direction
 */
template <Direction direction>
Aligner1D<direction> fromDictionary(Dictionary dic, const AxisNames& axis_names) {
    return fromDictionary<direction>(dic, axis_names[direction]);
}

template <Direction direction>
inline Aligner1D<direction> fromXML(const XMLReader& reader, const std::string& axis_name) {
     return fromDictionary<direction>([&](const std::string& s) { return reader.getAttribute<double>(s); }, axis_name);
}

template <Direction direction>
inline Aligner1D<direction> fromXML(const XMLReader& reader, const AxisNames& axis_names) {
    return fromXML<direction>(reader, axis_names[direction]);
}

/**
 * Construct 3D aligner from single string
 *
 * @param str string which describes 3d aligner
 * @return pointer to the constructed aligner
 */
//Aligner2D<Primitive<3>::DIRECTION_LONG, Primitive<3>::DIRECTION_TRAN>* alignerFromString(std::string str);

/**
 * Construct 3D aligner from two strings describing alignment in two directions
 *
 * @param str1 string which describes 2D aligner in the first direction
 * @param str2 string which describes 2D aligner in the second direction
 * @return pointer to the constructed 3D aligner
 */
template <Direction direction1, Direction direction2, typename Axes>
inline Aligner2D<direction1, direction2> fromDictionary(Dictionary dic, const Axes& axes) {
    Aligner1D<direction1> a1 = fromDictionary<direction1>(dic, axes);
    if (a1.isNull()) throw Exception("No aligner for axis%1% defined.", direction1);
    Aligner1D<direction2> a2 = fromDictionary<direction2>(dic, axes);
    if (a2.isNull()) throw Exception("No aligner for axis%1% defined.", direction2);
    return Aligner2D<direction1, direction2>(a1, a2);
}

}   // namespace align
}   // namespace plask

#endif // PLASK__GEOMETRY_ALIGN_H
