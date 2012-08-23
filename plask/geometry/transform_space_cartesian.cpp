#include "transform_space_cartesian.h"

#include <algorithm>
#include "reader.h"

namespace plask {

void Extrusion::setLength(double new_length) {
    if (length == new_length) return;
    length = new_length;
    fireChanged(Event::RESIZE);
}

bool Extrusion::include(const DVec& p) const {
    return canBeInside(p) && getChild()->include(childVec(p));
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

void Extrusion::getBoundingBoxesToVec(const GeometryElement::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    std::vector<ChildBox> c = getChild()->getBoundingBoxes(predicate, path);
    std::transform(c.begin(), c.end(), std::back_inserter(dest),
                   [&](const ChildBox& r) { return parentBox(r); });
}

std::vector< plask::shared_ptr< const plask::GeometryElement > > Extrusion::getLeafs() const {
    return getChild()->getLeafs();
}

shared_ptr<GeometryElementTransform<3, Extrusion::ChildType>> Extrusion::shallowCopy() const {
    return shared_ptr<GeometryElementTransform<3, Extrusion::ChildType>>(new Extrusion(getChild(), length));
}

GeometryElement::Subtree Extrusion::getPathsTo(const DVec& point) const {
    return GeometryElement::Subtree::extendIfNotEmpty(this, getChild()->getPathsTo(childVec(point)));
}

void Extrusion::writeXML(XMLWriter::Element& parent_xml_element, const WriteXMLCallback& write_cb, AxisNames axes) const {
    XMLWriter::Element tag = write_cb.makeTag(parent_xml_element, *this, "extrusion", axes);
    tag.attr("length", length);
    if (auto c = getChild()) c->writeXML(tag, write_cb, axes);
}

shared_ptr<GeometryElement> read_cartesianExtend(GeometryReader& reader) {
    double length = reader.source.requireAttribute<double>("length");
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    return make_shared<Extrusion>(reader.readExactlyOneChild<typename Extrusion::ChildType>(), length);
}

static GeometryReader::RegisterElementReader cartesianExtend2D_reader("extrusion", read_cartesianExtend);

}   // namespace plask
