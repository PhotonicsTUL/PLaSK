#include "space_changer_cartesian.h"

#include <algorithm>

namespace plask {

bool CartesianExtend::inside(const Vec& p) const {
    return canBeInside(p) && child().inside(childVec(p));
}

bool CartesianExtend::intersect(const Rect& area) const {
    return canIntersect(area) && child().intersect(childRect(area));
}

CartesianExtend::Rect CartesianExtend::getBoundingBox() const {
    return parentRect(child().getBoundingBox());
}

std::shared_ptr<Material> CartesianExtend::getMaterial(const Vec& p) const {
    return canBeInside(p) ? child().getMaterial(childVec(p)) : nullptr;
}

std::vector<CartesianExtend::Rect> CartesianExtend::getLeafsBoundingBoxes() const {
    std::vector<ChildRect> c = child().getLeafsBoundingBoxes();
    std::vector<Rect> result(c.size());
    std::transform(c.begin(), c.end(), result.begin(), [&](const ChildRect& r) { return parentRect(r); });
    return result;
}

}   // namespace plask
