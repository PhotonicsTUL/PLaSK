#ifndef PLASK__SPACE_CHANGER_CARTESIAN_H
#define PLASK__SPACE_CHANGER_CARTESIAN_H

#include "element.h"
#include "calculation_space.h"

namespace plask {

struct CartesianExtend: public GeometryElementChangeSpace<3, 2>, public CalculationSpace {

    double length;

    ///Size of calculation space.
    Rect2d spaceSize;

    virtual bool inside(const Vec& p) const;

    virtual bool intersect(const Rect& area) const;

    virtual Rect getBoundingBox() const;

    virtual std::shared_ptr<Material> getMaterial(const Vec& p) const;

    virtual std::vector<Rect> getLeafsBoundingBoxes() const;

private:
    ///@return true only if p can be inside this, false if for sure its not inside
    bool canBeInside(const Vec& p) const { return 0.0 <= p.c2 || p.c2 <= length; }

    ///@return true only if area can intersect this, false if for sure its not intersect
    bool canIntersect(const Rect& area) const { return !(area.lower.c2 > length || area.upper.c2 < 0.0); }

    ///@return p without last coordinate
    static ChildVec childVec(const Vec& p) { return ChildVec(p.c0, p.c1); }

    static ChildRect childRect(const Rect& r) { return ChildRect(childVec(r.lower), childVec(r.upper)); }

    static Vec parentVec(const ChildVec& p, double c2) { return Vec(p.c0, p.c1, c2); }

    Rect parentRect(const ChildRect& r) const { return Rect(parentVec(r.lower, 0.0), parentVec(r.upper, length)); }
};

}   // namespace plask

#endif // SPACE_CHANGER_CARTESIAN_H
