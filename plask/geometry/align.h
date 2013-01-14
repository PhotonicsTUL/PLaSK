#ifndef PLASK__GEOMETRY_ALIGN_H
#define PLASK__GEOMETRY_ALIGN_H

/** @file
This file includes aligners.
*/

#include "transform.h"
#include <memory>   //unique_ptr
#include <boost/lexical_cast.hpp>
#include "../utils/xml.h"

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


template <Direction direction> struct AxisAligner;

/**
 * Helper which allow to implement base class for aligners which work in one direction.
 * Don't use it directly, use AxisAligner instead.
 * @tparam _direction direction of activity
 * @see AxisAligner
 */
template <Direction _direction>
struct AxisAlignerBase: Printable {

    /// Direction of activity.
    static const Direction direction = _direction;

    /// Coordinate to which this aligner align.
    double coordinate;

    /**
     * Construct new aligner.
     * @param coordinate coordinate to which this aligner align.
     */
    AxisAlignerBase(double coordinate): coordinate(coordinate) {}

    virtual ~AxisAlignerBase() {}

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
     * Clone this aligner.
     * @return copy of this aligner, construted using operator @c new, caller must delete this copy after use
     */
    virtual AxisAligner<direction>* clone() const = 0;

    /**
     * Clone this aligner.
     * @return copy of this aligner, construted using operator @c new, and wrapped by std::unique_ptr
     */
    std::unique_ptr< AxisAligner<direction> > cloneUnique() const { return std::unique_ptr< AxisAligner<direction> >(clone()); }

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

/**
 * Base class for one direction aligners (in 2D and 3D spaces).
 */
template <Direction direction>
struct AxisAligner: public AxisAlignerBase<direction> {

    enum { direction2D = DirectionTo2D<direction>::value };

    using AxisAlignerBase<direction>::align;

    AxisAligner(double coordinate): AxisAlignerBase<direction>(coordinate) {}

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
    virtual double align(Translation<2>& toAlign) const {
        if (this->useBounds())
            return align(toAlign, toAlign.getChild()->getBoundingBox());
        else
            return toAlign.translation[direction2D] = this->getAlign(0.0, 0.0);
    }
};

template <>
struct AxisAligner<Primitive<3>::DIRECTION_LONG>: public AxisAlignerBase<Primitive<3>::DIRECTION_LONG> {
    AxisAligner<Primitive<3>::DIRECTION_LONG>(double coordinate): AxisAlignerBase<Primitive<3>::DIRECTION_LONG>(coordinate) {}
};

/**
 * Alginer which place zero of object in constant, chosen place.
 */
template <Direction direction>
struct PositionAxisAligner: public AxisAligner<direction> {

    PositionAxisAligner(double translation): AxisAligner<direction>(translation) {}

    virtual double getAlign(double low, double hi) const {
        return this->coordinate;
    }

    bool useBounds() const { return false; }

    PositionAxisAligner* clone() const { return new PositionAxisAligner(this->coordinate); }

    virtual void print(std::ostream& out) const { out << "align object position along axis " << direction << " to " << this->coordinate; }

    virtual std::string key(const AxisNames& axis_names) const { return axis_names[direction]; }
};

/**
 * Base class for two directions aligner in 3d space.
 */
template <Direction _direction1, Direction _direction2>
struct Aligner3D: Printable {

    static_assert(_direction1 != _direction2, "Wrong Aligner3D template parameters, two different directions are required.");

    virtual ~Aligner3D() {}

    static const Direction direction1 = _direction1, direction2 = _direction2;

    /**
     * Set object translation in directions of aligner activity.
     *
     * This version is called if caller knows child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     * @param childBoundingBox bounding box of object to align
     */
    virtual void align(Translation<3>& toAlign, const Box3D& childBoundingBox) const = 0;

    /**
     * Set object translation in directions of aligner activity.
     *
     * This version is called if caller doesn't know child bounding box.
     * @param toAlign trasnlation to set, should have child, which is an object to align
     */
    virtual void align(Translation<3>& toAlign) const {
        align(toAlign, toAlign.getChild()->getBoundingBox());
    }

    /**
     * Clone this aligner.
     * @return copy of this aligner, construted using operator @c new, caller must delete this copy after use
     */
    virtual Aligner3D<direction1, direction2>* clone() const = 0;

    /**
     * Clone this aligner.
     * @return copy of this aligner, construted using operator @c new, and wrapped by std::unique_ptr
     */
    std::unique_ptr< Aligner3D<direction1, direction2> > cloneUnique() const { return std::unique_ptr< Aligner3D<direction1, direction2> >(clone()); }

    /**
     * Get aligner as dictionary
     * \param axis_names name of axes
     * \return string:double map representing the aligner
     */
    virtual std::map<std::string,double> asDict(const AxisNames& axis_names) const = 0;

    /**
     * Write this aligner to XML.
     * @param dest tag where attributes describing this should be appended
     * @param axis_names name of axes
     */
    virtual void writeToXML(XMLElement& dest, const AxisNames& axis_names) const = 0;

};

/*template <Direction direction1, Direction direction2>
struct TranslationAligner3D: public Aligner3D<direction1, direction2> {

    ///Translations in aligner directions.
    double dir1translation, dir2translation;

    TranslationAligner3D(double dir1translation, double dir2translation): dir1translation(dir1translation), dir2translation(dir2translation) {}

    virtual void align(Translation<3>& toAlign, const Box3D&) const {
        align(toAlign);
    }

    virtual void align(Translation<3>& toAlign) const {
        toAlign.translation[direction1] = dir1translation;
        toAlign.translation[direction2] = dir2translation;
    }

    virtual TranslationAligner3D<direction1, direction2>* clone() const {
        return new TranslationAligner3D<direction1, direction2>(dir1translation, dir2translation);
    }

    virtual std::string strFirstDirection() const { return boost::lexical_cast<std::string>(dir1translation); }
    virtual std::string strSecondDirection() const { return boost::lexical_cast<std::string>(dir2translation); }
};*/

/**
 * Aligner 3d which compose and use two 2d aligners.
 */
template <Direction direction1, Direction direction2>
class ComposeAligner3D: public Aligner3D<direction1, direction2> {

    AxisAligner<direction1>* dir1aligner;
    AxisAligner<direction2>* dir2aligner;

public:

    ComposeAligner3D(const AxisAligner<direction1>& dir1aligner, const AxisAligner<direction2>& dir2aligner)
        : dir1aligner(dir1aligner.clone()), dir2aligner(dir2aligner.clone()) {}

    ComposeAligner3D(const ComposeAligner3D<direction1, direction2>& toCopy)
        : dir1aligner(toCopy.dir1aligner->clone()), dir2aligner(toCopy.dir2aligner->clone()) {}

    ComposeAligner3D(const ComposeAligner3D<direction2, direction1>& toCopy)
        : dir1aligner(toCopy.dir2aligner->clone()), dir2aligner(toCopy.dir1aligner->clone()) {}

    ComposeAligner3D(ComposeAligner3D<direction1, direction2>&& toMove)
        : dir1aligner(toMove.dir1aligner), dir2aligner(toMove.dir2aligner) {
        toMove.dir1aligner = 0; toMove.dir2aligner = 0;
    }

    ComposeAligner3D(ComposeAligner3D<direction2, direction1>&& toMove)
        : dir1aligner(toMove.dir2aligner), dir2aligner(toMove.dir1aligner) {
        toMove.dir1aligner = 0; toMove.dir2aligner = 0;
    }

    virtual ~ComposeAligner3D() { delete dir1aligner; delete dir2aligner; }

    virtual void align(Translation<3>& toAlign, const Box3D& childBoundingBox) const {
         toAlign.translation[direction1] =
                 dir1aligner->getAlign(childBoundingBox.lower[direction1], childBoundingBox.upper[direction1]);
         toAlign.translation[direction2] =
                 dir2aligner->getAlign(childBoundingBox.lower[direction2], childBoundingBox.upper[direction2]);
    }

    virtual void align(Translation<3>& toAlign) const {
        if (dir1aligner->useBounds() || dir2aligner->useBounds())
            align(toAlign, toAlign.getChild()->getBoundingBox());
        else {
            toAlign.translation[direction1] = dir1aligner->getAlign(0.0, 0.0);
            toAlign.translation[direction2] = dir2aligner->getAlign(0.0, 0.0);
        }
    }

    virtual ComposeAligner3D<direction1, direction2>* clone() const {
        return new ComposeAligner3D<direction1, direction2>(*this);
    }

    virtual void print(std::ostream& out) const { out << *dir1aligner << ", " << *dir2aligner; }

    virtual std::map<std::string,double> asDict(const AxisNames& axis_names) const {
        std::map<std::string,double> dict;
        dict[dir1aligner->key(axis_names)] = dir1aligner->coordinate;
        dict[dir2aligner->key(axis_names)] = dir2aligner->coordinate;
        return dict;
    }

    virtual void writeToXML(XMLElement& dest, const AxisNames& axis_names) const {
        dir1aligner->writeToXML(dest, axis_names);
        dir2aligner->writeToXML(dest, axis_names);
    }

};

inline ComposeAligner3D<Primitive<3>::Direction(0), Primitive<3>::Direction(1)> operator&(const AxisAligner<Primitive<3>::Direction(0)>& dir1aligner, const AxisAligner<Primitive<3>::Direction(1)>& dir2aligner) {
    return ComposeAligner3D<Primitive<3>::Direction(0), Primitive<3>::Direction(1)>(dir1aligner, dir2aligner);
}

inline ComposeAligner3D<Primitive<3>::Direction(0), Primitive<3>::Direction(1)> operator&(const AxisAligner<Primitive<3>::Direction(1)>& dir1aligner, const AxisAligner<Primitive<3>::Direction(0)>& dir2aligner) {
    return ComposeAligner3D<Primitive<3>::Direction(0), Primitive<3>::Direction(1)>(dir2aligner, dir1aligner);
}

inline ComposeAligner3D<Primitive<3>::Direction(0), Primitive<3>::Direction(2)> operator&(const AxisAligner<Primitive<3>::Direction(0)>& dir1aligner, const AxisAligner<Primitive<3>::Direction(2)>& dir2aligner) {
    return ComposeAligner3D<Primitive<3>::Direction(0), Primitive<3>::Direction(2)>(dir1aligner, dir2aligner);
}

inline ComposeAligner3D<Primitive<3>::Direction(0), Primitive<3>::Direction(2)> operator&(const AxisAligner<Primitive<3>::Direction(2)>& dir1aligner, const AxisAligner<Primitive<3>::Direction(0)>& dir2aligner) {
    return ComposeAligner3D<Primitive<3>::Direction(0), Primitive<3>::Direction(2)>(dir2aligner, dir1aligner);
}

inline ComposeAligner3D<Primitive<3>::Direction(1), Primitive<3>::Direction(2)> operator&(const AxisAligner<Primitive<3>::Direction(1)>& dir1aligner, const AxisAligner<Primitive<3>::Direction(2)>& dir2aligner) {
    return ComposeAligner3D<Primitive<3>::Direction(1), Primitive<3>::Direction(2)>(dir1aligner, dir2aligner);
}

inline ComposeAligner3D<Primitive<3>::Direction(1), Primitive<3>::Direction(2)> operator&(const AxisAligner<Primitive<3>::Direction(2)>& dir1aligner, const AxisAligner<Primitive<3>::Direction(1)>& dir2aligner) {
    return ComposeAligner3D<Primitive<3>::Direction(1), Primitive<3>::Direction(2)>(dir2aligner, dir1aligner);
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
struct AxisAlignerImpl: public AxisAligner<direction> {

    AxisAlignerImpl(double coordinate): AxisAligner<direction>(coordinate) {}

    virtual double getAlign(double low, double hi) const {
        return strategy(low, hi, this->coordinate);
    }

    virtual AxisAlignerImpl<direction, strategy, name_tag>* clone() const {
        return new AxisAlignerImpl<direction, strategy, name_tag>(this->coordinate);
    }

    virtual void print(std::ostream& out) const { out << "align " << name_tag::value << " to " << this->coordinate; }

    virtual std::string key(const AxisNames& axis_names) const { return name_tag::value; }
};

}   // namespace details

//2d trans. aligners:
typedef details::AxisAlignerImpl<Primitive<3>::DIRECTION_TRAN, details::lowToCoordinate, details::LEFT> Left;
typedef details::AxisAlignerImpl<Primitive<3>::DIRECTION_TRAN, details::hiToCoordinate, details::RIGHT> Right;
typedef details::AxisAlignerImpl<Primitive<3>::DIRECTION_TRAN, details::centerToCoordinate, details::TRAN_CENTER> TranCenter;
typedef details::AxisAlignerImpl<Primitive<3>::DIRECTION_TRAN, details::centerToCoordinate, details::TRAN_CENTER> Center;
typedef PositionAxisAligner<Primitive<3>::DIRECTION_TRAN> Tran;

//2d lon. aligners:
typedef details::AxisAlignerImpl<Primitive<3>::DIRECTION_LONG, details::hiToCoordinate, details::FRONT> Front;
typedef details::AxisAlignerImpl<Primitive<3>::DIRECTION_LONG, details::lowToCoordinate, details::BACK> Back;
typedef details::AxisAlignerImpl<Primitive<3>::DIRECTION_LONG, details::centerToCoordinate, details::LON_CENTER> LongCenter;
typedef PositionAxisAligner<Primitive<3>::DIRECTION_LONG> Long;

//2d vert. aligners:
typedef details::AxisAlignerImpl<Primitive<3>::DIRECTION_VERT, details::lowToCoordinate, details::BOTTOM> Bottom;
typedef details::AxisAlignerImpl<Primitive<3>::DIRECTION_VERT, details::hiToCoordinate, details::TOP> Top;
typedef details::AxisAlignerImpl<Primitive<3>::DIRECTION_VERT, details::centerToCoordinate, details::VERT_CENTER> VertCenter;
typedef PositionAxisAligner<Primitive<3>::DIRECTION_VERT> Vert;

//3d lon/tran aligners:
/*typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::hiToCoordinate, details::FRONT, Primitive<3>::DIRECTION_TRAN, details::lowToCoordinate, details::LEFT> FrontLeft;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::hiToCoordinate, details::FRONT, Primitive<3>::DIRECTION_TRAN, details::hiToCoordinate, details::RIGHT> FrontRight;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::hiToCoordinate, details::FRONT, Primitive<3>::DIRECTION_TRAN, details::centerToCoordinate, details::CENTER> FrontCenter;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::lowToCoordinate, details::BACK, Primitive<3>::DIRECTION_TRAN, details::lowToCoordinate, details::LEFT> BackLeft;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::lowToCoordinate, details::BACK, Primitive<3>::DIRECTION_TRAN, details::hiToCoordinate, details::RIGHT> BackRight;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::lowToCoordinate, details::BACK, Primitive<3>::DIRECTION_TRAN, details::centerToCoordinate, details::CENTER> BackCenter;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::centerToCoordinate, details::CENTER, Primitive<3>::DIRECTION_TRAN, details::lowToCoordinate, details::LEFT> CenterLeft;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::centerToCoordinate, details::CENTER, Primitive<3>::DIRECTION_TRAN, details::hiToCoordinate, details::RIGHT> CenterRight;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::centerToCoordinate, details::CENTER, Primitive<3>::DIRECTION_TRAN, details::centerToCoordinate, details::CENTER> CenterCenter;
typedef TranslationAligner3D<Primitive<3>::DIRECTION_LONG, Primitive<3>::DIRECTION_TRAN> LonTran;*/
//typedef ComposeAligner3D<DIR3D_LON, DIR3D_TRAN> NFLR;
//TODO mixed variants

typedef std::function<boost::optional<double>(const std::string& name)> Dictionary;

namespace details {
    std::unique_ptr<AxisAligner<Primitive<3>::DIRECTION_TRAN>> transAlignerFromDictionary(Dictionary dic, const std::string& axis_name);
    std::unique_ptr<AxisAligner<Primitive<3>::DIRECTION_LONG>> lonAlignerFromDictionary(Dictionary dic, const std::string& axis_name);
    std::unique_ptr<AxisAligner<Primitive<3>::DIRECTION_VERT>> vertAlignerFromDictionary(Dictionary dic, const std::string& axis_name);
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
std::unique_ptr<AxisAligner<direction>> fromDictionary(Dictionary dic, const std::string& axis_name);

template <>
inline std::unique_ptr<AxisAligner<Primitive<3>::DIRECTION_TRAN>> fromDictionary<Primitive<3>::DIRECTION_TRAN>(Dictionary dic, const std::string& axis_name) {
    return details::transAlignerFromDictionary(dic, axis_name);
}

template <>
inline std::unique_ptr<AxisAligner<Primitive<3>::DIRECTION_LONG>> fromDictionary<Primitive<3>::DIRECTION_LONG>(Dictionary dic, const std::string& axis_name) {
    return details::lonAlignerFromDictionary(dic, axis_name);
}

template <>
inline std::unique_ptr<AxisAligner<Primitive<3>::DIRECTION_VERT>> fromDictionary<Primitive<3>::DIRECTION_VERT>(Dictionary dic, const std::string& axis_name) {
    return details::vertAlignerFromDictionary(dic, axis_name);
}

/**
 * Construct 2d aligner in given direction from dictionary.
 *
 * Throw excpetion if @p dic includes information about multiple aligners in given @p direction.
 * @param dictionary dictionary which can describes 2D aligner
 * @param axis_names names of axes
 * @return parsed aligner or nullptr if no information found
 * @tparam direction direction
 */
template <Direction direction>
std::unique_ptr<AxisAligner<direction>> fromDictionary(Dictionary dic, const AxisNames& axis_names) {
    return fromDictionary<direction>(dic, axis_names[direction]);
}

template <Direction direction>
inline std::unique_ptr<AxisAligner<direction>> fromXML(const XMLReader& reader, const std::string& axis_name) {
     return fromDictionary<direction>([&](const std::string& s) { return reader.getAttribute<double>(s); }, axis_name);
}

template <Direction direction>
inline std::unique_ptr<AxisAligner<direction>> fromXML(const XMLReader& reader, const AxisNames& axis_names) {
    return fromXML<direction>(reader, axis_names[direction]);
}

/**
 * Construct 3d aligner from single string
 *
 * @param str string which describes 3d aligner
 * @return pointer to the constructed aligner
 */
//Aligner3D<Primitive<3>::DIRECTION_LONG, Primitive<3>::DIRECTION_TRAN>* alignerFromString(std::string str);

/**
 * Construct 3d aligner from two strings describing alignment in two directions
 *
 * @param str1 string which describes 2D aligner in the first direction
 * @param str2 string which describes 2D aligner in the second direction
 * @return pointer to the constructed 3D aligner
 */
template <Direction direction1, Direction direction2>
inline ComposeAligner3D<direction1, direction2> fromDictionary(Dictionary dic) {
    std::unique_ptr<AxisAligner<direction1>> a1 = fromDictionary<direction1>(dic);
    if (!a1) throw Exception("No aligner axis%1% defined.", direction1);
    std::unique_ptr<AxisAligner<direction1>> a2 = fromDictionary<direction2>(dic);
    if (!a2) throw Exception("No aligner axis%1% defined.", direction2);
    return ComposeAligner3D<direction1, direction2>(a1->release(), a2->release());
}

}   // namespace align
}   // namespace plask

#endif // PLASK__GEOMETRY_ALIGN_H
