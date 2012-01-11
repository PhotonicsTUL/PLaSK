#ifndef PLASK__SPACE_CHANGER_CARTESIAN_H
#define PLASK__SPACE_CHANGER_CARTESIAN_H

#include "element.h"
#include "calculation_space.h"

namespace plask {

/**
 * Represent 3d geometry element which are extend of 2d element (child) in lon direction.
 */
struct CartesianExtend: public GeometryElementChangeSpace<3, 2>/*, public CalculationSpace*/ {

    double length;

    //Size of calculation space.
    //Rect2d spaceSize;

    explicit CartesianExtend(shared_ptr<ChildType> child, double length): GeometryElementChangeSpace<3, 2>(child), length(length) {}

    explicit CartesianExtend(double length/*, Rect2d spaceSize*/): length(length)/*, spaceSize(spaceSize)*/ {}

    virtual bool inside(const DVec& p) const;

    virtual bool intersect(const Rect& area) const;

    virtual Rect getBoundingBox() const;

    virtual shared_ptr<Material> getMaterial(const DVec& p) const;

    virtual std::vector<Rect> getLeafsBoundingBoxes() const;

private:
    ///@return true only if p can be inside this, false if for sure its not inside
    bool canBeInside(const DVec& p) const { return 0.0 <= p.lon || p.lon <= length; }

    ///@return true only if area can intersect this, false if for sure its not intersect
    bool canIntersect(const Rect& area) const { return !(area.lower.lon > length || area.upper.lon < 0.0); }

    /**
     * Convert vector from this space (3d) to child space (2d).
     * @param p vector in space of this (3d)
     * @return @a p without lon coordinate, vector in space of child (2d)
     */
    static ChildVec childVec(const DVec& p) { return ChildVec(p.tran, p.up); }

    /**
     * Convert rectangle from this space (3d) to child space (2d).
     * @param r rectangle in space of this (3d)
     * @return @a r rectangle in space of child (2d)
     */
    static ChildRect childRect(const Rect& r) { return ChildRect(childVec(r.lower), childVec(r.upper)); }

    /**
     * Convert vector from child space (2d) to this space (3d).
     * @param p vector in space of child (2d), without lon coordinate
     * @param lon lon coordinate
     * @return vector in space of this (3d)
     */
    static DVec parentVec(const ChildVec& p, double lon) { return DVec(lon, p.tran, p.up); }

    /**
     * Convert rectangle from child space (2d) to this space (3d).
     * @param r rectangle in space of child (2d)
     * @return rectangle in space of this (3d), in lon direction: from 0.0 to @a length
     */
    Rect parentRect(const ChildRect& r) const { return Rect(parentVec(r.lower, 0.0), parentVec(r.upper, length)); }
};

}   // namespace plask

#endif // SPACE_CHANGER_CARTESIAN_H
