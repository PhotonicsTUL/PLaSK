#include "transform_space_cartesian.h"

#include <algorithm>
#include "reader.h"

namespace plask {

void Extrusion::setLength(double new_length) {
    if (length == new_length) return;
    length = new_length;
    fireChanged(Event::RESIZE);
}

bool Extrusion::includes(const DVec& p) const {
    return canBeInside(p) && getChild()->includes(childVec(p));
}

bool Extrusion::intersects(const Box& area) const {
    return canIntersect(area) && getChild()->intersects(childBox(area));
}

Extrusion::Box Extrusion::getBoundingBox() const {
    return parentBox(getChild()->getBoundingBox());
}

shared_ptr<Material> Extrusion::getMaterial(const DVec& p) const {
    return canBeInside(p) ? getChild()->getMaterial(childVec(p)) : shared_ptr<Material>();
}

void Extrusion::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    std::vector<ChildBox> c = getChild()->getBoundingBoxes(predicate, path);
    std::transform(c.begin(), c.end(), std::back_inserter(dest),
                   [&](const ChildBox& r) { return parentBox(r); });
}

std::vector< plask::shared_ptr< const plask::GeometryObject > > Extrusion::getLeafs() const {
    return getChild()->getLeafs();
}

shared_ptr<GeometryObjectTransform<3, Extrusion::ChildType>> Extrusion::shallowCopy() const {
    return shared_ptr<GeometryObjectTransform<3, Extrusion::ChildType>>(new Extrusion(getChild(), length));
}

GeometryObject::Subtree Extrusion::getPathsAt(const DVec& point, bool all) const {
    return GeometryObject::Subtree::extendIfNotEmpty(this, getChild()->getPathsAt(childVec(point), all));
}

void Extrusion::writeXMLAttr(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const {
    dest_xml_object.attr("length", length);
}

// void Extrusion::extractToVec(const GeometryObject::Predicate &predicate, std::vector< shared_ptr<const GeometryObjectD<3> > >&dest, const PathHints *path) const {
//     if (predicate(*this)) {
//         dest.push_back(static_pointer_cast< const GeometryObjectD<3> >(this->shared_from_this()));
//         return;
//     }
//     std::vector< shared_ptr<const GeometryObjectD<2> > > child_res = getChild()->extract(predicate, path);
//     for (shared_ptr<const GeometryObjectD<2>>& c: child_res)
//         dest.emplace_back(new Extrusion(const_pointer_cast<GeometryObjectD<2>>(c), this->length));
// }

shared_ptr<GeometryObject> read_cartesianExtend(GeometryReader& reader) {
    double length = reader.source.requireAttribute<double>("length");
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    return make_shared<Extrusion>(reader.readExactlyOneChild<typename Extrusion::ChildType>(), length);
}

static GeometryReader::RegisterObjectReader cartesianExtend2D_reader(Extrusion::NAME, read_cartesianExtend);

}   // namespace plask
