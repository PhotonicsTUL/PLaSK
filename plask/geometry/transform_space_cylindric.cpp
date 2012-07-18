#include "transform_space_cylindric.h"

namespace plask {

bool Revolution::include(const GeometryElementD< 3 >::DVec& p) const {
    return getChild()->include(childVec(p));
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

GeometryElement::Subtree Revolution::getPathsTo(const DVec& point) const {
    return GeometryElement::Subtree::extendIfNotEmpty(this, getChild()->getPathsTo(childVec(point)));
}

Box2D Revolution::childBox(const plask::Box3D& r) {
    Box2D result(childVec(r.lower), childVec(r.upper));
    result.fix();
    return result;
}

Box3D Revolution::parentBox(const ChildBox& r) {
    return Box3D(
            vec(-r.upper.tran, -r.upper.tran, r.lower.up),
            vec(r.upper.tran,  r.upper.tran,  r.upper.up)
           );
}


}   // namespace plask
