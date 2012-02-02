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

std::vector<CartesianExtend::Rect> CartesianExtend::getLeafsBoundingBoxes() const {
    std::vector<ChildRect> c = getChild()->getLeafsBoundingBoxes();
    std::vector<Rect> result(c.size());
    std::transform(c.begin(), c.end(), result.begin(), [&](const ChildRect& r) { return parentRect(r); });
    return result;
}

std::vector< boost::shared_ptr< const plask::GeometryElement > > CartesianExtend::getLeafs() const {
    return getChild()->getLeafs();
}

shared_ptr<GeometryElement> read_cartesianExtend(GeometryReader& reader) {
    double length = XML::requireAttr<double>(reader.source, "length");
    //TODO read space size
    return shared_ptr<GeometryElement>(new CartesianExtend(reader.readExactlyOneChild<typename CartesianExtend::ChildType>(), length));
}

GeometryReader::RegisterElementReader cartesianExtend2d_reader("extend", read_cartesianExtend);

}   // namespace plask
