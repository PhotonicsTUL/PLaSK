#ifndef PLASK__TRANSFORM_SPACE_CYLINDRIC_H
#define PLASK__TRANSFORM_SPACE_CYLINDRIC_H

#include "transform.h"

namespace plask {

/**
 * Represent 3D geometry object which is an effect of revolving a 2D object (child) around the up axis.
 *
 * Child should have getBoundingBox().lower.tran() >= 0. When it doesn't have, it is implicitly clipped.
 * @ingroup GEOMETRY_OBJ
 */
struct PLASK_API Revolution: public GeometryObjectTransformSpace<3, 2> {

    /**
     * @param child object to revolve
     * @param auto_clip if false child must have getBoundingBox().lower.tran() >= 0, if true it will be cliped
     */
    Revolution(shared_ptr<ChildType> child = shared_ptr<ChildType>(), bool auto_clip = false): GeometryObjectTransformSpace<3, 2>(child) {
        if (!auto_clip && childIsClipped())
            throw Exception("Child of Revolution must have bouding box with possitive tran. coordinates (when auto clipping is off).");
    }

    static constexpr const char* NAME = "revolution";

    virtual std::string getTypeName() const override;

    virtual bool contains(const DVec& p) const override;

    //TODO good but unused
    //virtual bool intersects(const Box& area) const;

    virtual shared_ptr<Material> getMaterial(const DVec& p) const override;

    virtual Box fromChildCoords(const typename ChildType::Box& child_bbox) const override;

    virtual shared_ptr<GeometryObjectTransform<3, GeometryObjectD<2> > > shallowCopy() const override;

    using GeometryObjectTransformSpace<3, 2>::getPathsTo;

    virtual GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const override;

    void getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path = 0) const override;

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

    /**
     * Check if child can be clipped.
     * @return @c true if child has getBoundingBox().lower.tran() < 0
     */
    bool childIsClipped() const;

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
