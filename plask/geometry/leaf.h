#ifndef PLASK__GEOMETRY_LEAF_H
#define PLASK__GEOMETRY_LEAF_H

/** @file
This file includes geometry elements leafs classes.
*/

#include "element.h"

namespace plask {

/**
Represent figure which, depends from @a dim is:
- for dim = 2 - rectangle,
- for dim = 3 - cuboid.
Block is filled with one material.
@tparam dim number of dimensions
*/
template <int dim>
struct Block: public GeometryElementLeaf<dim> {

    typedef typename GeometryElementLeaf<dim>::DVec DVec;
    typedef typename GeometryElementLeaf<dim>::Rect Rect;

    /**
     * Size and upper corner of block. Lower corner is zeroed vector.
     */
    DVec size;

    /**
     * Create block.
     * @param size size/upper corner of block
     * @param material block material
     */
    explicit Block(const DVec& size = Primitive<dim>::ZERO_VEC, shared_ptr<Material> material = shared_ptr<Material>())
        : GeometryElementLeaf<dim>(material), size(size) {}

    virtual Rect getBoundingBox() const {
        return Rect(Primitive<dim>::ZERO_VEC, size);
    }

    virtual bool inside(const DVec& p) const {
        return getBoundingBox().inside(p);
    }

    virtual bool intersect(const Rect& area) const {
        return getBoundingBox().intersect(area);
    }

};



}    // namespace plask

#endif // PLASK__GEOMETRY_LEAF_H
