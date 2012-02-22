#include "space_changer_cylindric.h"

namespace plask {

bool SpaceChangerCylindric::inside(const GeometryElementD< 3 >::DVec& p) const {
    return getChild()->inside(childVec(p));
}

bool SpaceChangerCylindric::intersect(const Rect& area) const {
    return getChild()->intersect(childRect(area));
}

SpaceChangerCylindric::Rect SpaceChangerCylindric::getBoundingBox() const {
    return parentRect(getChild()->getBoundingBox());
}

shared_ptr<Material> SpaceChangerCylindric::getMaterial(const DVec& p) const {
    return getChild()->getMaterial(childVec(p));
}

std::vector<SpaceChangerCylindric::Rect> SpaceChangerCylindric::getLeafsBoundingBoxes() const {
    std::vector<ChildRect> c = getChild()->getLeafsBoundingBoxes();
    std::vector<Rect> result(c.size());
    std::transform(c.begin(), c.end(), result.begin(), [&](const ChildRect& r) { return parentRect(r); });
    return result;
}

Box2d SpaceChangerCylindric::childRect(const plask::Box3d& r) {
    Box2d result(childVec(r.lower), childVec(r.upper));
    result.fix();
    return result;
}

Box3d SpaceChangerCylindric::parentRect(const ChildRect& r) {
    return Box3d(
            vec(-r.upper.tran, -r.upper.tran, r.lower.up),
            vec(r.upper.tran,  r.upper.tran,  r.upper.up)
           );
}


}   // namespace plask
