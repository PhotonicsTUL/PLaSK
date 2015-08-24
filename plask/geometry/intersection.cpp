#include "intersection.h"

#include "reader.h"

namespace plask {

template <int dim>
shared_ptr<Material> Intersection<dim>::getMaterial(const typename Intersection<dim>::DVec &p) const {
    return (this->hasChild() && this->inEnvelop(p)) ?
                this->_child->getMaterial(p) :
                shared_ptr<Material>();
}

template <int dim>
bool Intersection<dim>::contains(const typename Intersection<dim>::DVec &p) const {
    return this->hasChild() && (inEnvelop(p)) && this->_child->contains(p);
}

template <int dim>
GeometryObject::Subtree Intersection<dim>::getPathsAt(const Intersection<dim>::DVec &point, bool all) const
{
    if (this->hasChild() && inEnvelop(point))
        return GeometryObject::Subtree::extendIfNotEmpty(this, this->_child->getPathsAt(point, all));
    else
        return GeometryObject::Subtree();
}

template <int dim>
typename Intersection<dim>::Box Intersection<dim>::fromChildCoords(const typename Intersection<dim>::ChildType::Box &child_bbox) const
{
    if (envelope)
        return envelope->getBoundingBox().intersection(child_bbox);
    else
        return child_bbox;
}

template <int dim>
void Intersection<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->getBoundingBox());
        return;
    }
    if (!this->hasChild()) return;
    std::vector<Box> result = this->_child->getBoundingBoxes(predicate, path);
    dest.reserve(dest.size() + result.size());
    if (envelope) {
        Box clipBox = envelope->getBoundingBox();
        for (Box& r: result) {
            r.makeIntersection(clipBox);
            dest.push_back(r);
        }
    } else {
        for (Box& r: result)
            dest.push_back(r);
    }
}

template <int dim>
void Intersection<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
    this->_getNotChangedPositionsToVec(predicate, dest, path);
}

template <int dim>
shared_ptr<GeometryObjectTransform<dim> > Intersection<dim>::shallowCopy() const {
    return copyShallow();
}

template <int dim>
void Intersection<dim>::writeXMLChildren(XMLWriter::Element &dest_xml_object, GeometryObject::WriteXMLCallback &write_cb, const AxisNames &axes) const {
    if (auto child = getChild()) {  //TODO work without child, maybe change the sequence child <-> envelope?
        child->writeXML(dest_xml_object, write_cb, axes);
        if (envelope) envelope->writeXML(dest_xml_object, write_cb, axes);
    }
}

template <int dim>
shared_ptr<GeometryObject> read_Intersection(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, dim == 2 ? PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D : PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    shared_ptr< Intersection<dim> > intersection = make_shared<Intersection<dim>>();
    reader.source.requireTag();
    intersection->setChild(reader.readObject<typename Intersection<dim>::ChildType>());
    if (reader.source.requireTagOrEnd()) {
        GeometryReader::RevertMaterialsAreRequired enableShapeOnlyMode(reader, false);
        intersection->envelope = reader.readObject<typename Intersection<dim>::ChildType>();
        reader.source.requireTagEnd();
    }
    return intersection;
}

static GeometryReader::RegisterObjectReader Intersection2D_reader(Intersection<2>::NAME, read_Intersection<2>);
static GeometryReader::RegisterObjectReader Intersection3D_reader(Intersection<3>::NAME, read_Intersection<3>);

template struct PLASK_API Intersection<2>;
template struct PLASK_API Intersection<3>;

}   // namespace plask
