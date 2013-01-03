#ifndef PLASK__GEOMETRY_ALIGN_H
#define PLASK__GEOMETRY_ALIGN_H

/** @file
This file includes aligners.
*/

#include "transform.h"
#include <memory>   //unique_ptr
#include <boost/lexical_cast.hpp>

namespace plask {

namespace align {

/**
 * Directions of aligners activity, same as vec<3, T> directions.
 */
typedef Primitive<3>::Direction Direction;

/// Convert Direction to 2D vector direction
template <Direction direction>
struct DirectionTo2D {
    //static_assert(false, "given 3D direction can be convert to vector 2D direction");
};

template <>
struct DirectionTo2D<Primitive<3>::DIRECTION_TRAN> {
    enum { value = 0 };
};

template <>
struct DirectionTo2D<Primitive<3>::DIRECTION_VERT> {
    enum { value = 1 };
};


template <Direction direction> struct Aligner2D;

/**
 * Base class for aligners which work in one direction.
 * @tparam _direction direction of activity
 * @see Aligner2D
 */
template <Direction _direction>
struct OneDirectionAligner {

    ///Direction of activity.
    static const Direction direction = _direction;

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
    virtual OneDirectionAligner<direction>* clone() const = 0;

    /**
     * Clone this aligner.
     * @return copy of this aligner, construted using operator @c new, and wrapped by std::unique_ptr
     */
    std::unique_ptr< OneDirectionAligner<direction> > cloneUnique() const { return std::unique_ptr< Aligner2D<direction> >(clone()); }

    /**
     * Get string representation of this aligner.
     * @return string representation of this aligner
     */
    virtual std::string str() const = 0;

    /// Virtual destructor. Do nothing.
    virtual ~OneDirectionAligner() {}

};

/**
 * Base class for one direction aligner in 2D space.
 */
template <Direction direction>
struct Aligner2D: public OneDirectionAligner<direction> {

    enum { direction2D = DirectionTo2D<direction>::value };

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

    /**
     * Clone this aligner.
     * @return copy of this aligner, construted using operator @c new, caller must delete this copy after use
     */
    virtual Aligner2D<direction>* clone() const = 0;

    /**
     * Clone this aligner.
     * @return copy of this aligner, construted using operator @c new, and wrapped by std::unique_ptr
     */
    std::unique_ptr< Aligner2D<direction> > cloneUnique() const { return std::unique_ptr< Aligner2D<direction> >(clone()); }

};

template <>
struct Aligner2D<Primitive<3>::DIRECTION_LONG>: public OneDirectionAligner<Primitive<3>::DIRECTION_LONG> {

    /**
     * Clone this aligner.
     * @return copy of this aligner, construted using operator @c new, caller must delete this copy after use
     */
    virtual Aligner2D<direction>* clone() const = 0;

    /**
     * Clone this aligner.
     * @return copy of this aligner, construted using operator @c new, and wrapped by std::unique_ptr
     */
    std::unique_ptr< Aligner2D<direction> > cloneUnique() const { return std::unique_ptr< Aligner2D<direction> >(clone()); }

};

/**
 * Alginer which place object in constant place.
 */
template <Direction direction>
struct TranslationAligner2D: public Aligner2D<direction> {

    /// Translation of aligned object in aligner activity direction.
    double translation;

    TranslationAligner2D(double translation): translation(translation) {}

    virtual double getAlign(double low, double hi) const {
        return translation;
    }

    bool useBounds() const { return false; }

    TranslationAligner2D* clone() const { return new TranslationAligner2D(translation); }

    virtual std::string str() const { return boost::lexical_cast<std::string>(translation); }
};

/**
 * Base class for two directions aligner in 3d space.
 */
template <Direction _direction1, Direction _direction2>
struct Aligner3D {

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

    virtual std::string strFirstDirection() const = 0;
    virtual std::string strSecondDirection() const = 0;

};

template <Direction direction1, Direction direction2>
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
};

/**
 * Aligner 3d which compose and use two 2d aligners.
 */
template <Direction direction1, Direction direction2>
class ComposeAligner3D: public Aligner3D<direction1, direction2> {

    Aligner2D<direction1>* dir1aligner;
    Aligner2D<direction2>* dir2aligner;

public:

    ComposeAligner3D(const Aligner2D<direction1>& dir1aligner, const Aligner2D<direction2>& dir2aligner)
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

    ~ComposeAligner3D() { delete dir1aligner; delete dir2aligner; }

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

    virtual std::string strFirstDirection() const { return dir1aligner->str(); }
    virtual std::string strSecondDirection() const { return dir2aligner->str(); }

};

template <Direction direction1, Direction direction2>
inline ComposeAligner3D<direction1, direction2> operator&(const Aligner2D<direction1>& dir1aligner, const Aligner2D<direction2>& dir2aligner) {
    return ComposeAligner3D<direction1, direction2>(dir1aligner, dir2aligner);
}

namespace details {

typedef double alignStrategy(double lo, double hi);
inline double lowToZero(double lo, double hi) { return -lo; }
inline double hiToZero(double lo, double hi) { return -hi; }
inline double centerToZero(double lo, double hi) { return -(lo+hi)/2.0; }

struct LEFT { static constexpr const char* value = "left"; };
struct RIGHT { static constexpr const char* value = "right"; };
struct FRONT { static constexpr const char* value = "front"; };
struct BACK { static constexpr const char* value = "back"; };
struct CENTER { static constexpr const char* value = "center"; };

template <Direction direction, alignStrategy strategy, typename name_tag>
struct Aligner2DImpl: public Aligner2D<direction> {

    virtual double getAlign(double low, double hi) const {
        return strategy(low, hi);
    }

    virtual Aligner2DImpl<direction, strategy, name_tag>* clone() const {
        return new Aligner2DImpl<direction, strategy, name_tag>();
    }

    virtual std::string str() const { return name_tag::value; }
};

template <Direction direction1, alignStrategy strategy1, typename str_tag1, Direction direction2, alignStrategy strategy2, typename str_tag2>
struct Aligner3DImpl: public Aligner3D<direction1, direction2> {

    virtual void align(Translation<3>& toAlign, const Box3D& childBoundingBox) const {
        toAlign.translation[direction1] = strategy1(childBoundingBox.lower[direction1], childBoundingBox.upper[direction1]);
        toAlign.translation[direction2] = strategy2(childBoundingBox.lower[direction2], childBoundingBox.upper[direction2]);
    }

    virtual Aligner3DImpl<direction1, strategy1, str_tag1, direction2, strategy2, str_tag2>* clone() const {
        return new Aligner3DImpl<direction1, strategy1, str_tag1, direction2, strategy2, str_tag2>();
    }

    virtual std::string strFirstDirection() const { return str_tag1::value; }
    virtual std::string strSecondDirection() const { return str_tag2::value; }
};

}   // namespace details

//2d trans. aligners:
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_TRAN, details::lowToZero, details::LEFT> Left;
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_TRAN, details::hiToZero, details::RIGHT> Right;
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_TRAN, details::centerToZero, details::CENTER> TranCenter;
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_TRAN, details::centerToZero, details::CENTER> Center;
typedef TranslationAligner2D<Primitive<3>::DIRECTION_TRAN> Tran;

//2d lon. aligners:
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_LONG, details::hiToZero, details::FRONT> Front;
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_LONG, details::lowToZero, details::BACK> Back;
typedef details::Aligner2DImpl<Primitive<3>::DIRECTION_LONG, details::centerToZero, details::CENTER> LonCenter;
typedef TranslationAligner2D<Primitive<3>::DIRECTION_LONG> Lon;

//3d lon/tran aligners:
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::hiToZero, details::FRONT, Primitive<3>::DIRECTION_TRAN, details::lowToZero, details::LEFT> FrontLeft;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::hiToZero, details::FRONT, Primitive<3>::DIRECTION_TRAN, details::hiToZero, details::RIGHT> FrontRight;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::hiToZero, details::FRONT, Primitive<3>::DIRECTION_TRAN, details::centerToZero, details::CENTER> FrontCenter;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::lowToZero, details::BACK, Primitive<3>::DIRECTION_TRAN, details::lowToZero, details::LEFT> BackLeft;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::lowToZero, details::BACK, Primitive<3>::DIRECTION_TRAN, details::hiToZero, details::RIGHT> BackRight;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::lowToZero, details::BACK, Primitive<3>::DIRECTION_TRAN, details::centerToZero, details::CENTER> BackCenter;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::centerToZero, details::CENTER, Primitive<3>::DIRECTION_TRAN, details::lowToZero, details::LEFT> CenterLeft;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::centerToZero, details::CENTER, Primitive<3>::DIRECTION_TRAN, details::hiToZero, details::RIGHT> CenterRight;
typedef details::Aligner3DImpl<Primitive<3>::DIRECTION_LONG, details::centerToZero, details::CENTER, Primitive<3>::DIRECTION_TRAN, details::centerToZero, details::CENTER> CenterCenter;
typedef TranslationAligner3D<Primitive<3>::DIRECTION_LONG, Primitive<3>::DIRECTION_TRAN> LonTran;
//typedef ComposeAligner3D<DIR3D_LON, DIR3D_TRAN> NFLR;
//TODO mixed variants

namespace details {
    Aligner2D<Primitive<3>::DIRECTION_TRAN>* transAlignerFromString(std::string str);
    Aligner2D<Primitive<3>::DIRECTION_LONG>* lonAlignerFromString(std::string str);
}

/**
 * Construct 2d aligner in given direction from string.
 * @param str string which describes 2d aligner
 * @tparam direction direction
 */
template <Direction direction>
Aligner2D<direction>* fromStr(const std::string& str);

template <>
inline Aligner2D<Primitive<3>::DIRECTION_TRAN>* fromStr<Primitive<3>::DIRECTION_TRAN>(const std::string& str) { return details::transAlignerFromString(str); }

template <>
inline Aligner2D<Primitive<3>::DIRECTION_LONG>* fromStr<Primitive<3>::DIRECTION_LONG>(const std::string& str) { return details::lonAlignerFromString(str); }

template <Direction direction>
inline std::unique_ptr<Aligner2D<direction>> fromStrUnique(const std::string& str) {
     return std::unique_ptr<Aligner2D<direction>>(fromStr<direction>(str));
}


/**
 * Construct 3d aligner from single string
 *
 * @param str string which describes 3d aligner
 * @return pointer to the constructed aligner
 **/
Aligner3D<Primitive<3>::DIRECTION_LONG, Primitive<3>::DIRECTION_TRAN>* alignerFromString(std::string str);

/**
 * Construct 3d aligner from two strings describing alignment in two directions
 *
 * @param str1 string which describes 2d aligner in the first direction
 * @param str2 string which describes 2d aligner in the second direction
 * @return pointer to the constructed 3d aligner
 **/
template <Direction direction1, Direction direction2>
inline ComposeAligner3D<direction1, direction2> fromStr(const std::string& str1, const std::string& str2) {
    std::unique_ptr<Aligner2D<direction1>> a1(fromStr<direction1>(str1));
    std::unique_ptr<Aligner2D<direction2>> a2(fromStr<direction2>(str2));
    return ComposeAligner3D<direction1, direction2>(*a1, *a2);
}

}   // namespace align
}   // namespace plask

#endif // PLASK__GEOMETRY_ALIGN_H
