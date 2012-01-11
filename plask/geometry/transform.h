#ifndef PLASK__GEOMETRY_TRANSFORM_H
#define PLASK__GEOMETRY_TRANSFORM_H

#include "element.h"

namespace plask {

/**
 * Represent geometry element equal to its child translated by vector.
 */
template <int dim>
struct Translation: public GeometryElementTransform<dim> {

    ///Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryElementTransform<dim>::DVec DVec;
    
    ///Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryElementTransform<dim>::Rect Rect;
    
    using GeometryElementTransform<dim>::getChild;

    /**
     * Translation vector.
     */
    DVec translation;

    /**
     * @param child child geometry element, element to translate
     * @param translation translation
     */
    explicit Translation(shared_ptr< GeometryElementD<dim> > child = shared_ptr< GeometryElementD<dim> >(), const DVec& translation = Primitive<dim>::ZERO_VEC)
        : GeometryElementTransform<dim>(child), translation(translation) {}

    virtual Rect getBoundingBox() const {
        return getChild()->getBoundingBox().translated(translation);
    }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        return getChild()->getMaterial(p-translation);
    }

    virtual bool inside(const DVec& p) const {
        return getChild()->inside(p-translation);
    }

    virtual bool intersect(const Rect& area) const {
        return getChild()->intersect(area.translated(-translation));
    }

    virtual std::vector<Rect> getLeafsBoundingBoxes() const {
        std::vector<Rect> result = getChild()->getLeafsBoundingBoxes();
        DVec inv_tr = - translation;
        for (Rect& r: result) r.translate(inv_tr);
        return result;
    }

};

}       // namespace plask

#endif // PLASK__GEOMETRY_TRANSFORM_H
