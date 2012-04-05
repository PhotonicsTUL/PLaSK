#include "transform_space_cylindric.h"

namespace plask {

bool Revolution::inside(const GeometryElementD< 3 >::DVec& p) const {
    return getChild()->inside(childVec(p));
}

bool Revolution::intersect(const Box& area) const {
    return getChild()->intersect(childBox(area));
}

Revolution::Box Revolution::getBoundingBox() const {
    return parentBox(getChild()->getBoundingBox());
}

shared_ptr<Material> Revolution::getMaterial(const DVec& p) const {
    return getChild()->getMaterial(childVec(p));
}

std::vector<Revolution::Box> Revolution::getLeafsBoundingBoxes() const {
    std::vector<ChildBox> c = getChild()->getLeafsBoundingBoxes();
    std::vector<Box> result(c.size());
    std::transform(c.begin(), c.end(), result.begin(), [&](const ChildBox& r) { return parentBox(r); });
    return result;
}

Box2d Revolution::childBox(const plask::Box3d& r) {
    Box2d result(childVec(r.lower), childVec(r.upper));
    result.fix();
    return result;
}

Box3d Revolution::parentBox(const ChildBox& r) {
    return Box3d(
            vec(-r.upper.tran, -r.upper.tran, r.lower.up),
            vec(r.upper.tran,  r.upper.tran,  r.upper.up)
           );
}


}   // namespace plask
