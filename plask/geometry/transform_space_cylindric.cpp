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

void Revolution::getBoundingBoxesToVec(const GeometryElement::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    std::vector<ChildBox> c = getChild()->getBoundingBoxes(predicate, path);
    std::transform(c.begin(), c.end(), std::back_inserter(dest),
                   [&](const ChildBox& r) { return parentBox(r); });
}

shared_ptr<GeometryElementTransform< 3, GeometryElementD<2> > > Revolution::shallowCopy() const {
    return make_shared<Revolution>(this->getChild());
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
