#ifndef PLASK__GEOMETRY_ALIGN_H
#define PLASK__GEOMETRY_ALIGN_H

#include "transform.h"

namespace plask {

namespace align {

enum DIRECTION_2D {
    DIR2D_TRAN, 
    DIR2D_UP
};

enum DIRECTION_3D {
    DIR3D_LON, 
    DIR3D_TRAN, 
    DIR3D_UP
};

/**
 * Base class for one direction aligner in 2d space.
 */
template <DIRECTION_2D _direction>
struct Aligner2d {
    
    static const DIRECTION_2D direction = _direction;
    
    //virtual double getAlign(double low, double hi);
    
    //This version is called if caller knwo bounding box.
    virtual double align(const Translation<2>& toAlign, const Box2d& childBoundingBox) const = 0;
    
    virtual double align(const Translation<2>& toAlign) const {
        align(toAlign, toAlign.getChild()->getBoundingBox());
    }
    
    virtual Aligner2d<direction>* clone() const = 0;
    
    virtual ~Aligner2d() {}
    
};

template <DIRECTION_2D direction>
struct TranslationAligner2d: public Aligner2d<direction> {
    
    ///Translation in aligner direction
    double translation;
    
    TranslationAligner2d(double translation): translation(translation) {}
    
    virtual double align(const Translation<2>& toAlign, const Box2d& childBoundingBox) const {
        toAlign.translation.components[direction] = translation;
    }
    
    virtual double align(const Translation<2>& toAlign) const {
        toAlign.translation.components[direction] = translation;
    }
    
    virtual TranslationAligner2d<direction>* clone() const {
        return new TranslationAligner2d<direction>(translation);
    }
};

/**
 * Base class for two directions aligner in 3d space.
 */
template <DIRECTION_3D _direction1, DIRECTION_3D _direction2>
struct Aligner3d {
    
    static const DIRECTION_3D direction1 = _direction1, direction2 = _direction2;
    
    //This version is called if caller knwo bounding box.
    virtual double align(const Translation<3>& toAlign, const Box3d& childBoundingBox) const = 0;
    
    virtual double align(const Translation<3>& toAlign) const {
        align(toAlign, toAlign.getChild()->getBoundingBox());
    }
    
    virtual Aligner3d<direction1, direction2>* clone() const = 0;
    
    virtual ~Aligner3d() {}
    
};

template <DIRECTION_3D direction1, DIRECTION_3D direction2>
struct TranslationAligner3d: public Aligner3d<direction1, direction2> {
    
    ///Translations in aligner directions.
    double dir1translation, dir2translation;
    
    TranslationAligner3d(double dir1translation, double dir2translation): dir1translation(dir1translation), dir2translation(dir2translation) {}
    
    virtual double align(const Translation<2>& toAlign, const Box2d&) const {
        align(toAlign);
    }
    
    virtual double align(const Translation<2>& toAlign) const {
        toAlign.translation.components[direction1] = dir1translation;
        toAlign.translation.components[direction2] = dir2translation;
    }
    
    virtual TranslationAligner3d<direction1, direction2>* clone() const {
        return new TranslationAligner3d<direction1, direction2>(dir1translation, dir2translation);
    }
};

/**
 * Aligner 3d which compose and use two 2d aligners. 
 */
/*template <DIRECTION_3D direction1, DIRECTION_3D direction2>
class ComposeAligner3d: public Aligner3d<direction1, direction2> {
    
    Aligner2d<direction1>* dir1aligner;
    Aligner2d<direction2>* dir2aligner;
    
public:
    
    ComposeAligner3d(const Aligner2d<direction1>& dir1aligner, const Aligner2d<direction2>& dir2aligner)
        : dir1aligner(dir1aligner.clone()), dir2aligner(dir2aligner.clone()) {}
    
    ~ComposeAligner3d() { delete dir1aligner; delete dir2aligner; }
    
    virtual double align(const Translation<3>& toAlign, const Box3d& childBoundingBox) const {
        
    }
    
};*/

namespace details {

typedef double alignStrategy(double lo, double hi);
inline double lowToZero(double lo, double hi) { return -lo; }
inline double hiToZero(double lo, double hi) { return -hi; }

template <DIRECTION_2D direction, alignStrategy strategy>
struct Aligner2dImpl: public Aligner2d<direction> {
    
    virtual double align(const Translation<2>& toAlign, const Box2d& childBoundingBox) const {
        toAlign.translation.components[direction] = strategy(childBoundingBox.lower.components[direction], childBoundingBox.upper.components[direction]);
    }
    
    virtual Aligner2dImpl<direction, strategy>* clone() const {
        return new Aligner2dImpl<direction, strategy>();
    }
};

template <DIRECTION_3D direction1, alignStrategy strategy1, DIRECTION_3D direction2, alignStrategy strategy2>
struct Aligner3dImpl: public Aligner3d<direction1, direction2> {
    
    virtual double align(const Translation<3>& toAlign, const Box3d& childBoundingBox) const {
        toAlign.translation.components[direction1] = strategy1(childBoundingBox.lower.components[direction1], childBoundingBox.upper.components[direction1]);
        toAlign.translation.components[direction2] = strategy2(childBoundingBox.lower.components[direction2], childBoundingBox.upper.components[direction2]);
    }
    
    virtual Aligner3dImpl<direction1, strategy1, direction2, strategy2>* clone() const {
        return new Aligner3dImpl<direction1, strategy1, direction2, strategy2>();
    }
};

}   // namespace details

//2d trasnlation aligners:
typedef details::Aligner2dImpl<DIR2D_TRAN, details::lowToZero> Left;
typedef details::Aligner2dImpl<DIR2D_TRAN, details::hiToZero> Right;
typedef TranslationAligner2d<DIR2D_TRAN> Tran;

//3d lon/tran aligners:
typedef details::Aligner3dImpl<DIR3D_LON, details::lowToZero, DIR3D_TRAN, details::lowToZero> NearLeft;
typedef details::Aligner3dImpl<DIR3D_LON, details::lowToZero, DIR3D_TRAN, details::hiToZero> NearRight;
typedef details::Aligner3dImpl<DIR3D_LON, details::hiToZero, DIR3D_TRAN, details::lowToZero> FarLeft;
typedef details::Aligner3dImpl<DIR3D_LON, details::hiToZero, DIR3D_TRAN, details::hiToZero> FarRight;
typedef TranslationAligner3d<DIR3D_LON, DIR3D_TRAN> LonTran;
//TODO mixed variants

}   // namespace align
}   // namespace plask

#endif // PLASK__GEOMETRY_ALIGN_H
