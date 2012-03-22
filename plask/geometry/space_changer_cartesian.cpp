#include "space_changer_cartesian.h"

#include <algorithm>
#include "reader.h"

namespace plask {

bool CartesianExtend::inside(const DVec& p) const {
    return canBeInside(p) && getChild()->inside(childVec(p));
}

bool CartesianExtend::intersect(const Rect& area) const {
    return canIntersect(area) && getChild()->intersect(childRect(area));
}

CartesianExtend::Rect CartesianExtend::getBoundingBox() const {
    return parentRect(getChild()->getBoundingBox());
}

shared_ptr<Material> CartesianExtend::getMaterial(const DVec& p) const {
    return canBeInside(p) ? getChild()->getMaterial(childVec(p)) : shared_ptr<Material>();
}

void CartesianExtend::getLeafsBoundingBoxesToVec(std::vector<Rect>& dest, const PathHints* path) const {
    std::vector<ChildRect> c = getChild()->getLeafsBoundingBoxes(path);
    std::transform(c.begin(), c.end(), std::back_inserter(dest),
                   [&](const ChildRect& r) { return parentRect(r); });
}

std::vector< boost::shared_ptr< const plask::GeometryElement > > CartesianExtend::getLeafs() const {
    return getChild()->getLeafs();
}

shared_ptr<GeometryElementTransform<3, CartesianExtend::ChildType>> CartesianExtend::shallowCopy() const {
    return shared_ptr<GeometryElementTransform<3, CartesianExtend::ChildType>>(new CartesianExtend(getChild(), length));
}

shared_ptr<GeometryElement> read_cartesianExtend(GeometryReader& reader) {
    double length = XML::requireAttr<double>(reader.source, "length");
    //TODO read space size
    return make_shared<CartesianExtend>(reader.readExactlyOneChild<typename CartesianExtend::ChildType>(), length);
}

static GeometryReader::RegisterElementReader cartesianExtend2d_reader("extend", read_cartesianExtend);

}   // namespace plask
