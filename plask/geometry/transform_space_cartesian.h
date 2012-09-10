#ifndef PLASK__TRANSFORM_SPACE_CARTESIAN_H
#define PLASK__TRANSFORM_SPACE_CARTESIAN_H

#include "transform.h"

namespace plask {

/**
 * Represent 3d geometry element which are extend of 2d element (child) in lon direction.
 */
struct Extrusion: public GeometryElementTransformSpace<3, 2>/*, public Geometry*/ {

    typedef GeometryElementTransformSpace<3, 2>::ChildType ChildType;

    double length;

    // Size of the calculation space.
    // spaceSize;

    explicit Extrusion(shared_ptr<ChildType> child, double length): GeometryElementTransformSpace<3,2>(child), length(length) {}

    explicit Extrusion(double length = 0.0/*,  spaceSize*/): length(length)/*, spaceSize(spaceSize)*/ {}

    static constexpr const char* NAME = "extrusion";

    virtual std::string getTypeName() const { return NAME; }

    /**
     * Set length and inform observers.
     * @param new_length new length
     */
    void setLength(double new_length);

    virtual bool includes(const DVec& p) const;

    virtual bool intersects(const Box& area) const;

    virtual Box getBoundingBox() const;

    virtual shared_ptr<Material> getMaterial(const DVec& p) const;

    //virtual void getLeafsInfoToVec(std::vector<std::tuple<shared_ptr<const GeometryElement>, Box, DVec>>& dest, const PathHints* path = 0) const;

    virtual void getBoundingBoxesToVec(const GeometryElement::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const;

    virtual std::vector< plask::shared_ptr< const plask::GeometryElement > > getLeafs() const;

    virtual shared_ptr<GeometryElementTransform<3, ChildType>> shallowCopy() const;

    using GeometryElementTransformSpace<3, 2>::getPathsTo;

    GeometryElement::Subtree getPathsTo(const DVec& point) const;

    void writeXMLAttr(XMLWriter::Element &dest_xml_element, const AxisNames &axes) const;

private:
    /// @return true only if p can be inside this, false if for sure its not inside
    bool canBeInside(const DVec& p) const { return 0.0 <= p.lon() || p.lon() <= length; }

    /// @return true only if area can intersects this, false if for sure its not intersects
    bool canIntersect(const Box& area) const { return !(area.lower.lon() > length || area.upper.lon() < 0.0); }

    /**
     * Convert vector from this space (3d) to child space (2d).
     * @param p vector in space of this (3d)
     * @return @p p without lon coordinate, vector in space of child (2d)
     */
    static ChildVec childVec(const DVec& p) { return ChildVec(p.tran(), p.up()); }

    /**
     * Convert box from this space (3d) to child space (2d).
     * @param r box in space of this (3d)
     * @return @p r box in space of child (2d)
     */
    static ChildBox childBox(const Box& r) { return ChildBox(childVec(r.lower), childVec(r.upper)); }

    /**
     * Convert vector from child space (2d) to this space (3d).
     * @param p vector in space of child (2d), without lon coordinate
     * @param lon lon coordinate
     * @return vector in space of this (3d)
     */
    static DVec parentVec(const ChildVec& p, double lon) { return DVec(lon, p.tran(), p.up()); }

    /**
     * Convert box from child space (2d) to this space (3d).
     * @param r box in space of child (2d)
     * @return box in space of this (3d), in lon direction: from 0.0 to @p length
     */
    Box parentBox(const ChildBox& r) const { return Box(parentVec(r.lower, 0.0), parentVec(r.upper, length)); }
};

}   // namespace plask

#endif // TRANSFORM_SPACE_CARTESIAN_H
