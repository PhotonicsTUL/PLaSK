#ifndef PLASK__GEOMETRY_ALIGN_H
#define PLASK__GEOMETRY_ALIGN_H

#include "transform.h"

namespace plask {

namespace align {

/**
 * Directions of aligners activity, same as vec<3, T> directions.
 */
enum DIRECTION {
    DIRECTION_LON,
    DIRECTION_TRAN,
    DIRECTION_UP
};

template <DIRECTION direction> struct Aligner2d;

/**
 * Helper class used to for implementation of Aligner2d.
 * Don't use directly, use Aligner2d instead.
 * @tparam _direction direction of activity
 */
template <DIRECTION _direction>
struct Aligner2dBase {

    ///Direction of activity.
    static const DIRECTION direction = _direction;

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
     * @return copy of this aligner
     */
    virtual Aligner2d<direction>* clone() const = 0;

    ///Virtual destructor. Do nothing.
    virtual ~Aligner2dBase() {}

};

/**
 * Base class for one direction aligner in 2d space.
 * @tparam _direction direction of activity
 */
template <DIRECTION direction>
struct Aligner2d: public Aligner2dBase<direction> {};

template <>
struct Aligner2d<DIRECTION_TRAN>: public Aligner2dBase<DIRECTION_TRAN> {

    //This version is called if caller knows bounding box.
    inline double align(Translation<2>& toAlign, const Box2d& childBoundingBox) const {
        toAlign.translation.tran = getAlign(childBoundingBox.lower.tran, childBoundingBox.upper.tran);
    }

    virtual double align(Translation<2>& toAlign) const {
        if (useBounds())
            align(toAlign, toAlign.getChild()->getBoundingBox());
        else
            toAlign.translation.tran = getAlign(0.0, 0.0);
    }

};

template <DIRECTION direction>
struct TranslationAligner2d: public Aligner2d<direction> {
    
    ///Translation in aligner activity direction
    double translation;
    
    TranslationAligner2d(double translation): translation(translation) {}
    
    virtual double getAlign(double low, double hi) const {
        return translation;
    }
    
    bool useBounds() const { return false; }

    TranslationAligner2d* clone() const { return new TranslationAligner2d(translation); }
};

/**
 * Base class for two directions aligner in 3d space.
 */
template <DIRECTION _direction1, DIRECTION _direction2>
struct Aligner3d {
    
    static_assert(_direction1 != _direction2, "Wrong Aligner3d template parameters, two different directions are required.");
    
    virtual ~Aligner3d() {}
    
    static const DIRECTION direction1 = _direction1, direction2 = _direction2;
    
    //This version is called if caller knows bounding box.
    virtual void align(Translation<3>& toAlign, const Box3d& childBoundingBox) const = 0;
    
    virtual void align(Translation<3>& toAlign) const {
        align(toAlign, toAlign.getChild()->getBoundingBox());
    }
    
    virtual Aligner3d<direction1, direction2>* clone() const = 0;

};

template <DIRECTION direction1, DIRECTION direction2>
struct TranslationAligner3d: public Aligner3d<direction1, direction2> {
    
    ///Translations in aligner directions.
    double dir1translation, dir2translation;
    
    TranslationAligner3d(double dir1translation, double dir2translation): dir1translation(dir1translation), dir2translation(dir2translation) {}
    
    virtual void align(Translation<3>& toAlign, const Box3d&) const {
        align(toAlign);
    }
    
    virtual void align(Translation<3>& toAlign) const {
        toAlign.translation.components[direction1] = dir1translation;
        toAlign.translation.components[direction2] = dir2translation;
    }
    
    virtual TranslationAligner3d<direction1, direction2>* clone() const {
        return new TranslationAligner3d<direction1, direction2>(dir1translation, dir2translation);
    }
};

/*
 * Aligner 3d which compose and use two 2d aligners. 
 */
template <DIRECTION direction1, DIRECTION direction2>
class ComposeAligner3d: public Aligner3d<direction1, direction2> {
    
    Aligner2d<direction1>* dir1aligner;
    Aligner2d<direction2>* dir2aligner;
    
public:
    
    ComposeAligner3d(const Aligner2d<direction1>& dir1aligner, const Aligner2d<direction2>& dir2aligner)
        : dir1aligner(dir1aligner.clone()), dir2aligner(dir2aligner.clone()) {}
    
    ComposeAligner3d(const ComposeAligner3d<direction1, direction2>& toCopy)
        : dir1aligner(toCopy.dir1aligner->clone()), dir2aligner(toCopy.dir2aligner->clone()) {}
    
    ComposeAligner3d(const ComposeAligner3d<direction2, direction1>& toCopy)
        : dir1aligner(toCopy.dir2aligner->clone()), dir2aligner(toCopy.dir1aligner->clone()) {}
    
    ComposeAligner3d(ComposeAligner3d<direction1, direction2>&& toMove)
        : dir1aligner(toMove.dir1aligner), dir2aligner(toMove.dir2aligner) {
        toMove.dir1aligner = 0; toMove.dir2aligner = 0;
    }
    
    ComposeAligner3d(ComposeAligner3d<direction2, direction1>&& toMove)
        : dir1aligner(toMove.dir2aligner), dir2aligner(toMove.dir1aligner) {
        toMove.dir1aligner = 0; toMove.dir2aligner = 0;
    }
    
    ~ComposeAligner3d() { delete dir1aligner; delete dir2aligner; }
    
    virtual void align(Translation<3>& toAlign, const Box3d& childBoundingBox) const {
         toAlign.translation.components[direction1] =
                 dir1aligner->getAlign(childBoundingBox.lower.components[direction1], childBoundingBox.upper.components[direction1]);
         toAlign.translation.components[direction2] =
                 dir2aligner->getAlign(childBoundingBox.lower.components[direction2], childBoundingBox.upper.components[direction2]);
    }

    virtual void align(Translation<3>& toAlign) const {
        if (dir1aligner->useBounds() || dir2aligner->useBounds())
            align(toAlign, toAlign.getChild()->getBoundingBox());
        else {
            toAlign.translation.components[direction1] = dir1aligner->getAlign(0.0, 0.0);
            toAlign.translation.components[direction2] = dir2aligner->getAlign(0.0, 0.0);
        }
    }
    
    virtual ComposeAligner3d<direction1, direction2>* clone() const {
        return new ComposeAligner3d<direction1, direction2>(*this);
    }
    
};

template <DIRECTION direction1, DIRECTION direction2>
inline ComposeAligner3d<direction1, direction2> operator&(const Aligner2d<direction1>& dir1aligner, const Aligner2d<direction2>& dir2aligner) {
    return ComposeAligner3d<direction1, direction2>(dir1aligner, dir2aligner);
}

namespace details {

typedef double alignStrategy(double lo, double hi);
inline double lowToZero(double lo, double hi) { return -lo; }
inline double hiToZero(double lo, double hi) { return -hi; }
inline double centerToZero(double lo, double hi) { return -(lo+hi)/2.0; }

template <DIRECTION direction, alignStrategy strategy>
struct Aligner2dImpl: public Aligner2d<direction> {
    
    virtual double getAlign(double low, double hi) const {
        return strategy(low, hi);
    }
    
    virtual Aligner2dImpl<direction, strategy>* clone() const {
        return new Aligner2dImpl<direction, strategy>();
    }
};

template <DIRECTION direction1, alignStrategy strategy1, DIRECTION direction2, alignStrategy strategy2>
struct Aligner3dImpl: public Aligner3d<direction1, direction2> {
    
    virtual void align(Translation<3>& toAlign, const Box3d& childBoundingBox) const {
        toAlign.translation.components[direction1] = strategy1(childBoundingBox.lower.components[direction1], childBoundingBox.upper.components[direction1]);
        toAlign.translation.components[direction2] = strategy2(childBoundingBox.lower.components[direction2], childBoundingBox.upper.components[direction2]);
    }
    
    virtual Aligner3dImpl<direction1, strategy1, direction2, strategy2>* clone() const {
        return new Aligner3dImpl<direction1, strategy1, direction2, strategy2>();
    }
};

}   // namespace details

//2d trasnlation aligners:
typedef details::Aligner2dImpl<DIRECTION_TRAN, details::lowToZero> Left;
typedef details::Aligner2dImpl<DIRECTION_TRAN, details::hiToZero> Right;
typedef details::Aligner2dImpl<DIRECTION_TRAN, details::centerToZero> LRCenter;
typedef TranslationAligner2d<DIRECTION_TRAN> Tran;

//2d lon. aligners:
typedef details::Aligner2dImpl<DIRECTION_TRAN, details::lowToZero> Near;
typedef details::Aligner2dImpl<DIRECTION_TRAN, details::hiToZero> Far;
typedef details::Aligner2dImpl<DIRECTION_TRAN, details::centerToZero> NFCenter;
typedef TranslationAligner2d<DIRECTION_TRAN> Lon;

//3d lon/tran aligners:
typedef details::Aligner3dImpl<DIRECTION_LON, details::lowToZero, DIRECTION_TRAN, details::lowToZero> NearLeft;
typedef details::Aligner3dImpl<DIRECTION_LON, details::lowToZero, DIRECTION_TRAN, details::hiToZero> NearRight;
typedef details::Aligner3dImpl<DIRECTION_LON, details::lowToZero, DIRECTION_TRAN, details::centerToZero> NearLRCenter;
typedef details::Aligner3dImpl<DIRECTION_LON, details::hiToZero, DIRECTION_TRAN, details::lowToZero> FarLeft;
typedef details::Aligner3dImpl<DIRECTION_LON, details::hiToZero, DIRECTION_TRAN, details::hiToZero> FarRight;
typedef details::Aligner3dImpl<DIRECTION_LON, details::hiToZero, DIRECTION_TRAN, details::centerToZero> FarLRCenter;
typedef details::Aligner3dImpl<DIRECTION_LON, details::centerToZero, DIRECTION_TRAN, details::lowToZero> NFCenerLeft;
typedef details::Aligner3dImpl<DIRECTION_LON, details::centerToZero, DIRECTION_TRAN, details::hiToZero> NFCenerRight;
typedef details::Aligner3dImpl<DIRECTION_LON, details::centerToZero, DIRECTION_TRAN, details::centerToZero> NFCenerLRCenter;
typedef TranslationAligner3d<DIRECTION_LON, DIRECTION_TRAN> LonTran;
//typedef ComposeAligner3d<DIR3D_LON, DIR3D_TRAN> NFLR;
//TODO mixed variants

}   // namespace align
}   // namespace plask

#endif // PLASK__GEOMETRY_ALIGN_H
