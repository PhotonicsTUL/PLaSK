#include "transform_space_cartesian.h"

#include <algorithm>
#include "reader.h"

namespace plask {

bool Extrusion::inside(const DVec& p) const {
    return canBeInside(p) && getChild()->inside(childVec(p));
}

bool Extrusion::intersect(const Box& area) const {
    return canIntersect(area) && getChild()->intersect(childBox(area));
}

Extrusion::Box Extrusion::getBoundingBox() const {
    return parentBox(getChild()->getBoundingBox());
}

shared_ptr<Material> Extrusion::getMaterial(const DVec& p) const {
    return canBeInside(p) ? getChild()->getMaterial(childVec(p)) : shared_ptr<Material>();
}

void Extrusion::getLeafsBoundingBoxesToVec(std::vector<Box>& dest, const PathHints* path) const {
    std::vector<ChildBox> c = getChild()->getLeafsBoundingBoxes(path);
    std::transform(c.begin(), c.end(), std::back_inserter(dest),
                   [&](const ChildBox& r) { return parentBox(r); });
}

std::vector< boost::shared_ptr< const plask::GeometryElement > > Extrusion::getLeafs() const {
    return getChild()->getLeafs();
}

shared_ptr<GeometryElementTransform<3, Extrusion::ChildType>> Extrusion::shallowCopy() const {
    return shared_ptr<GeometryElementTransform<3, Extrusion::ChildType>>(new Extrusion(getChild(), length));
}

shared_ptr<GeometryElement> read_cartesianExtend(GeometryReader& reader) {
    double length = XML::requireAttr<double>(reader.source, "length");
    //TODO read space size
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    return make_shared<Extrusion>(reader.readExactlyOneChild<typename Extrusion::ChildType>(), length);
}

static GeometryReader::RegisterElementReader cartesianExtend2d_reader("extrusion", read_cartesianExtend);

}   // namespace plask
