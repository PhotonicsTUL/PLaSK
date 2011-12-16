#include "space_changer_cartesian.h"

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

}   // namespace plask
