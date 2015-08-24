#include "transform_space_cartesian.h"

#include <algorithm>
#include "reader.h"

namespace plask {

std::string Extrusion::getTypeName() const { return NAME; }

void Extrusion::setLength(double new_length) {
    if (length == new_length) return;
    length = new_length;
    fireChanged(Event::EVENT_RESIZE);
}

bool Extrusion::contains(const DVec& p) const {
    return (this->hasChild() && canBeInside(p)) && this->_child->contains(childVec(p));
}

/*bool Extrusion::intersects(const Box& area) const {
    return canIntersect(area) && getChild()->intersects(childBox(area));
}*/

shared_ptr<Material> Extrusion::getMaterial(const DVec& p) const {
    return (this->hasChild() && canBeInside(p)) ? this->_child->getMaterial(childVec(p)) : shared_ptr<Material>();
}

Extrusion::Box Extrusion::fromChildCoords(const Extrusion::ChildType::Box &child_bbox) const {
    return parentBox(child_bbox);
}

/*std::vector< plask::shared_ptr< const plask::GeometryObject > > Extrusion::getLeafs() const {
    return getChild()->getLeafs();
}*/

shared_ptr<GeometryObjectTransform<3, Extrusion::ChildType>> Extrusion::shallowCopy() const {
    return shared_ptr<GeometryObjectTransform<3, Extrusion::ChildType>>(new Extrusion(this->_child, length));
}

GeometryObject::Subtree Extrusion::getPathsAt(const DVec& point, bool all) const {
    if (this->hasChild() && canBeInside(point))
        return GeometryObject::Subtree::extendIfNotEmpty(this, getChild()->getPathsAt(childVec(point), all));
    else
        return GeometryObject::Subtree();
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
