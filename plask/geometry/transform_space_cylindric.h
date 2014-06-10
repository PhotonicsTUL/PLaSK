#ifndef PLASK__TRANSFORM_SPACE_CYLINDRIC_H
#define PLASK__TRANSFORM_SPACE_CYLINDRIC_H

#include "transform.h"

namespace plask {

/**
 * Represent 3D geometry object which is an effect of revolving a 2D object (child) around the up axis.
 *
 * Child must have getBoundingBox().lower.tran() >= 0.
 * @ingroup GEOMETRY_OBJ
 */
struct PLASK_API Revolution: public GeometryObjectTransformSpace<3, 2> {

    /**
     * @param child object to revolve, must have getBoundingBox().lower.tran() >= 0
     */
    Revolution(shared_ptr<ChildType> child = shared_ptr<ChildType>()): GeometryObjectTransformSpace<3, 2>(child) {}

    static constexpr const char* NAME = "revolution";

    virtual std::string getTypeName() const { return NAME; }

    virtual bool contains(const DVec& p) const;

    //TODO good but unused
    //virtual bool intersects(const Box& area) const;

    virtual Box getBoundingBox() const;

    virtual shared_ptr<Material> getMaterial(const DVec& p) const;

    virtual void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const;

    virtual shared_ptr<GeometryObjectTransform<3, GeometryObjectD<2> > > shallowCopy() const;

    using GeometryObjectTransformSpace<3, 2>::getPathsTo;

    virtual GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const;

    // virtual void extractToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObjectD<dim> > >& dest, const PathHints* = 0) const;

    /**
     * Convert vector @p v to space of child.
     * @param r vector in parent (this) space, i.e. in space where (0, 0, 0) is at center of base of cylinder
     * @return vector in child space
     */
    static Vec<2, double> childVec(const Vec<3, double>& v) {
        return rotateToLonTranAbs(v);
    }

    /*
     * Convert rectangle @p r to space of child.
     * @param r cuboid in parent (this) space
     * @return rectangle in child space
     */
    //static ChildBox childBox(const Box& r);

private:
    /**
     * Convert rectangle @p r to space of parent (this).
     * @param r rectangle in child space
     * @return cuboid in parent (this) space
     */
    static Box parentBox(const ChildBox& r);
};

}   // namespace plask

#endif // PLASK__TRANSFORM_SPACE_CYLINDRIC_H
