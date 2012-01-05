#ifndef PLASK__GEOMETRY_TRANSFORM_H
#define PLASK__GEOMETRY_TRANSFORM_H

#include "element.h"

namespace plask {

/**
 * Translate child by vector.
 */
template <int dim>
struct Translation: public GeometryElementTransform<dim> {

    typedef typename GeometryElementTransform<dim>::DVec DVec;
    typedef typename GeometryElementTransform<dim>::Rect Rect;
    using GeometryElementTransform<dim>::getChild;

    DVec translation;

    explicit Translation(GeometryElementD<dim>* child = 0, const DVec& translation = Primitive<dim>::ZERO_VEC)
        : GeometryElementTransform<dim>(child), translation(translation) {}

    explicit Translation(GeometryElementD<dim>& child, const DVec& translation = Primitive<dim>::ZERO_VEC)
        : GeometryElementTransform<dim>(&child), translation(translation) {}

    virtual Rect getBoundingBox() const {
        return getChild().getBoundingBox().translated(translation);
    }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        return getChild().getMaterial(p-translation);
    }

    virtual bool inside(const DVec& p) const {
        return getChild().inside(p-translation);
    }

    virtual bool intersect(const Rect& area) const {
        return getChild().intersect(area.translated(-translation));
    }

    virtual std::vector<Rect> getLeafsBoundingBoxes() const {
        std::vector<Rect> result = getChild().getLeafsBoundingBoxes();
        DVec inv_tr = - translation;
        for (Rect& r: result) r.translate(inv_tr);
        return result;
    }

};

}       // namespace plask

#endif // PLASK__GEOMETRY_TRANSFORM_H
