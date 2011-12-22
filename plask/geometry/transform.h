#ifndef PLASK__GEOMETRY_TRANSFORM_H
#define PLASK__GEOMETRY_TRANSFORM_H

#include "element.h"

namespace plask {

/**
 * Translate child by vector.
 */
template <int dim>
struct Translation: public GeometryElementTransform<dim> {

    typedef typename GeometryElementTransform<dim>::Vec Vec;
    typedef typename GeometryElementTransform<dim>::Rect Rect;
    using GeometryElementTransform<dim>::getChild;

    Vec translation;

    Translation(GeometryElementD<dim>* child, const Vec& translation): GeometryElementTransform<dim>(child), translation(translation) {}
    
    Translation(GeometryElementD<dim>& child, const Vec& translation): GeometryElementTransform<dim>(&child), translation(translation) {}

    virtual Rect getBoundingBox() const {
        return getChild().getBoundingBox().translated(translation);
    }
    
    virtual shared_ptr<Material> getMaterial(const Vec& p) const {
        return getChild().getMaterial(p-translation);
    }

    virtual bool inside(const Vec& p) const {
        return getChild().inside(p-translation);
    }

    virtual bool intersect(const Rect& area) const {
        return getChild().intersect(area.translated(-translation));
    }
    
    virtual std::vector<Rect> getLeafsBoundingBoxes() const {
        std::vector<Rect> result = getChild().getLeafsBoundingBoxes();
        Vec inv_tr = - translation;
        for (Rect& r: result) r.translate(inv_tr);
        return result;
    }

};

}       // namespace plask

#endif // PLASK__GEOMETRY_TRANSFORM_H
