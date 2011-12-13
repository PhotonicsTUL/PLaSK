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
    using GeometryElementTransform<dim>::child;

    Vec translation;

    Translation(GeometryElementD<dim>* child, const Vec& translation): GeometryElementTransform<dim>(child), translation(translation) {}

    virtual Rect getBoundingBox() {
        return child().getBoundingBox().translated(translation);
    }

    virtual bool inside(const Vec& p) const {
        return child().inside(p-translation);
    }

    virtual bool intersect(const Rect& area) const {
        return child().intersect(area.translated(-translation));
    }

};

}       // namespace plask

#endif // PLASK__GEOMETRY_TRANSFORM_H
