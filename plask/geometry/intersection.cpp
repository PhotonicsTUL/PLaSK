#include "intersection.h"

#include "reader.h"

namespace plask {

template <int dim>
typename Intersection<dim>::Box Intersection<dim>::getBoundingBox() const {
    return getChild()->getBoundingBox().intersection(clippingShape->getBoundingBox());
}

template <int dim>
shared_ptr<Material> Intersection<dim>::getMaterial(const typename Intersection<dim>::DVec &p) const {
    return clippingShape->contains(p) ? getChild()->getMaterial(p) : shared_ptr<Material>();
}

template <int dim>
bool Intersection<dim>::contains(const typename Intersection<dim>::DVec &p) const {
    return clippingShape->contains(p) && getChild()->contains(p);
}

template <int dim>
GeometryObject::Subtree Intersection<dim>::getPathsAt(const Intersection<dim>::DVec &point, bool all) const
{
    if (clippingShape->contains(point))
        return GeometryObject::Subtree::extendIfNotEmpty(this, getChild()->getPathsAt(point, all));
        else
            return GeometryObject::Subtree();
}

template <int dim>
void Intersection<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    std::vector<Box> result = getChild()->getBoundingBoxes(predicate, path);
    dest.reserve(dest.size() + result.size());
    Box clipBox = clippingShape->getBoundingBox();
    for (Box& r: result) {
        r.makeIntersection(clipBox);
        dest.push_back(r);
    }
}

template <int dim>
void Intersection<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    getChild()->getPositionsToVec(predicate, dest, path);
}

template <int dim>
shared_ptr<GeometryObjectTransform<dim> > Intersection<dim>::shallowCopy() const {
    return copyShallow();
}

template <int dim>
void Intersection<dim>::writeXMLChildren(XMLWriter::Element &dest_xml_object, GeometryObject::WriteXMLCallback &write_cb, const AxisNames &axes) const {
    if (auto child = getChild()) {
        child->writeXML(dest_xml_object, write_cb, axes);
        if (clippingShape) clippingShape->writeXML(dest_xml_object, write_cb, axes);
    }
}

template <int dim>
shared_ptr<GeometryObject> read_Intersection(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, dim == 2 ? PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D : PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    shared_ptr< Intersection<dim> > intersection = make_shared<Intersection<dim>>();
    intersection->setChild(reader.readObject<typename Intersection<dim>::ChildType>());
    {
        GeometryReader::RevertMaterialsAreRequired enableShapeOnlyMode(reader, false);
        intersection->clippingShape = reader.readObject<typename Intersection<dim>::ChildType>();
    }
    reader.source.requireTagEnd();
    return intersection;
}

static GeometryReader::RegisterObjectReader Intersection2D_reader(Intersection<2>::NAME, read_Intersection<2>);
static GeometryReader::RegisterObjectReader Intersection3D_reader(Intersection<3>::NAME, read_Intersection<3>);

template struct PLASK_API Intersection<2>;
template struct PLASK_API Intersection<3>;

}   // namespace plask
