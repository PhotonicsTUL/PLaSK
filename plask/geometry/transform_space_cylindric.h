#ifndef PLASK__TRANSFORM_SPACE_CYLINDRIC_H
#define PLASK__TRANSFORM_SPACE_CYLINDRIC_H

#include "transform.h"

namespace plask {

/**
 * Represent 3d geometry element which is an effect of revolving a 2d element (child) around the up axis.
 *
 * Child must have getBoundingBox().lower.tran >= 0.
 */
struct Revolution: public GeometryElementTransformSpace<3, 2> {

    /**
     * @param child element to revolve, must have getBoundingBox().lower.tran >= 0
     */
    Revolution(shared_ptr<ChildType> child): GeometryElementTransformSpace<3, 2>(child) {}

    virtual bool include(const DVec& p) const;

    virtual bool intersect(const Box& area) const;

    virtual Box getBoundingBox() const;

    virtual shared_ptr<Material> getMaterial(const DVec& p) const;

    virtual void getBoundingBoxesToVec(const GeometryElement::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const;

    virtual shared_ptr<GeometryElementTransform<3, GeometryElementD<2> > > shallowCopy() const;

    using GeometryElementTransformSpace<3, 2>::findPathsTo;

    virtual GeometryElement::Subtree findPathsTo(const DVec& point) const;

private:

    /**
     * Convert vector @p v to space of child.
     * @param r vector in parent (this) space
     * @return vector in child space
     */
    static Vec<2, double> childVec(const Vec<3, double>& v) {
        return vec(sqrt(v.lon*v.lon + v.tran*v.tran), v.up);
    }

    /**
     * Convert rectangle @p r to space of child.
     * @param r cuboid in parent (this) space
     * @return rectangle in child space
     */
    static ChildBox childBox(const Box& r);

    /**
     * Convert rectangle @p r to space of parent (this).
     * @param r rectangle in child space
     * @return cuboid in parent (this) space
     */
    static Box parentBox(const ChildBox& r);
};

}   // namespace plask

#endif // PLASK__TRANSFORM_SPACE_CYLINDRIC_H
