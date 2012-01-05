#include "space_changer_cartesian.h"

#include <algorithm>

namespace plask {

bool CartesianExtend::inside(const DVec& p) const {
    return canBeInside(p) && getChild().inside(childVec(p));
}

bool CartesianExtend::intersect(const Rect& area) const {
    return canIntersect(area) && getChild().intersect(childRect(area));
}

CartesianExtend::Rect CartesianExtend::getBoundingBox() const {
    return parentRect(getChild().getBoundingBox());
}

shared_ptr<Material> CartesianExtend::getMaterial(const DVec& p) const {
    return canBeInside(p) ? getChild().getMaterial(childVec(p)) : shared_ptr<Material>();
}

std::vector<CartesianExtend::Rect> CartesianExtend::getLeafsBoundingBoxes() const {
    std::vector<ChildRect> c = getChild().getLeafsBoundingBoxes();
    std::vector<Rect> result(c.size());
    std::transform(c.begin(), c.end(), result.begin(), [&](const ChildRect& r) { return parentRect(r); });
    return result;
}

}   // namespace plask
