#include "space.h"

#include <memory>
#include <cassert>

namespace plask {

void Geometry::setEdges(const std::function<plask::optional<std::string>(const std::string& s)>& borderValuesGetter, const AxisNames& axesNames, const MaterialsDB &materialsDB)
{
    plask::optional<std::string> v, v_lo, v_hi;
    v = borderValuesGetter("edges");
    if (v) setAllEdges(*edge::Strategy::fromStrUnique(*v, materialsDB));
    v = borderValuesGetter("planar");
    if (v) setPlanarEdges(*edge::Strategy::fromStrUnique(*v, materialsDB));
    for (int dir_nr = 0; dir_nr < 3; ++dir_nr) {
        std::string axis_name = axesNames[dir_nr];
        v = borderValuesGetter(axis_name);
        if (v) setEdges(Direction(dir_nr), *edge::Strategy::fromStrUnique(*v, materialsDB));
        v_lo = borderValuesGetter(axis_name + "-lo");
        if ((v = borderValuesGetter(alternativeDirectionName(dir_nr, 0)))) {
            if (v_lo) throw BadInput("setEdges", "Egde specified by both '{0}-lo' and '{1}'", axis_name, alternativeDirectionName(dir_nr, 0));
            else v_lo = v;
        }
        v_hi = borderValuesGetter(axis_name + "-hi");
        if ((v = borderValuesGetter(alternativeDirectionName(dir_nr, 1)))) {
            if (v_hi) throw BadInput("setEdges", "Egde specified by both '{0}-hi' and '{1}'", axis_name, alternativeDirectionName(dir_nr, 1));
            else v_hi = v;
        }
        try {
            if (v_lo && v_hi) {
                setEdges(Direction(dir_nr),  *edge::Strategy::fromStrUnique(*v_lo, materialsDB), *edge::Strategy::fromStrUnique(*v_hi, materialsDB));
            } else {
                if (v_lo) setEdge(Direction(dir_nr), false, *edge::Strategy::fromStrUnique(*v_lo, materialsDB));
                if (v_hi) setEdge(Direction(dir_nr), true, *edge::Strategy::fromStrUnique(*v_hi, materialsDB));
            }
        } catch (DimensionError) {
            throw BadInput("setEdges", "Axis '{0}' is not allowed for this space", axis_name);
        }
    }
}

void Geometry::storeEdgeInXML(XMLWriter::Element &dest_xml_object, Geometry::Direction direction, bool higher) const {
    const edge::Strategy& b = this->getEdge(direction, higher);
    if (b.type() != edge::Strategy::DEFAULT)
        dest_xml_object.attr(this->alternativeDirectionName(direction, higher), b.str());
}

template <int dim>
void GeometryD<dim>::onChildChanged(const GeometryObject::Event &evt) {
    if (evt.isResize()) cachedBoundingBox = getChild()->getBoundingBox();
    //comipler should optimized out dim == 2 condition checking
    fireChanged(evt.originalSource(), dim == 2 ? evt.flagsForParentWithChildrenWasChangedInformation() : evt.flagsForParent());
}

template <int dim>
void GeometryD<dim>::disconnectOnChildChanged() {
    connection_with_child.disconnect();
}

template <int dim>
void GeometryD<dim>::initNewChild() {
    disconnectOnChildChanged(); //disconnect old child, if any
    auto c3d = getObject3D();
    if (c3d) {
        if (c3d) connection_with_child = c3d->changedConnectMethod(this, &GeometryD<dim>::onChildChanged);
        auto c = getChildUnsafe();
        if (c) cachedBoundingBox = c->getBoundingBox();
    }
}

template <int dim>
int GeometryD<dim>::getDimensionsCount() const { return DIM; }

template <int dim>
shared_ptr<Material> GeometryD<dim>::getMaterial(const Vec<dim, double> &p) const {
    return getMaterialOrDefault(p);
}

template <int dim>
std::set<std::string> GeometryD<dim>::getRolesAt(const typename GeometryD<dim>::CoordsType &point, const PathHints *path) const {
    return getChild()->getRolesAt(point, path);
}

template <int dim>
std::set<std::string> GeometryD<dim>::getRolesAt(const typename GeometryD<dim>::CoordsType &point, const PathHints &path) const {
    return getChild()->getRolesAt(point, &path);
}

template <>
void GeometryD<2>::writeXMLAttr(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const {
    Geometry::writeXMLAttr(dest_xml_object, axes);
    dest_xml_object.attr("axes", axes.str());
    this->storeEdgeInXML(dest_xml_object, DIRECTION_TRAN, false);
    this->storeEdgeInXML(dest_xml_object, DIRECTION_TRAN, true);
    this->storeEdgeInXML(dest_xml_object, DIRECTION_VERT, false);
    this->storeEdgeInXML(dest_xml_object, DIRECTION_VERT, true);
}

template <>
void GeometryD<3>::writeXMLAttr(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const {
    Geometry::writeXMLAttr(dest_xml_object, axes);
    dest_xml_object.attr("axes", axes.str());
    for (int dir = 0; dir < 3; ++dir) {
        this->storeEdgeInXML(dest_xml_object, plask::Geometry::Direction(dir), false);
        this->storeEdgeInXML(dest_xml_object, plask::Geometry::Direction(dir), true);
    }
}

template class PLASK_API GeometryD<2>;
template class PLASK_API GeometryD<3>;

Geometry2DCartesian::Geometry2DCartesian(shared_ptr<Extrusion> extrusion)
    : extrusion(extrusion)
{
    initNewChild();
}

Geometry2DCartesian::Geometry2DCartesian(shared_ptr<GeometryObjectD<2>> childGeometry, double length)
    : extrusion(plask::make_shared<Extrusion>(childGeometry, length))
{
    initNewChild();
}

shared_ptr< GeometryObjectD<2> > Geometry2DCartesian::getChild() const {
    if (!extrusion) throw NoChildException();
    auto child = extrusion->getChild();
    if (!child) throw NoChildException();
    return child;
}

shared_ptr< GeometryObjectD<2> > Geometry2DCartesian::getChildUnsafe() const {
    return extrusion->getChild();
}

shared_ptr<Material> Geometry2DCartesian::getMaterial(const Vec<2, double> &p) const {
    Vec<2, double> r = p;
    shared_ptr<Material> material;

    bottomup.apply(cachedBoundingBox, r, material);
    if (material) return material;

    leftright.apply(cachedBoundingBox, r, material);
    if (material) return material;

    return getMaterialOrDefault(r);
}

void Geometry2DCartesian::setExtrusion(shared_ptr<Extrusion> extrusion) {
    if (this->extrusion == extrusion) return;
    this->extrusion = extrusion;
    this->initNewChild();
    fireChildrenChanged();
}

// Geometry2DCartesian* Geometry2DCartesian::getSubspace(const shared_ptr<GeometryObjectD<2>>& object, const PathHints* path, bool copyEdges) const {
//     auto shifts = getChild()->getObjectPositions(object, path);
//     // auto new_child = getChild()->getUniqueObjectInThisCoordinates(object, path);
//     // if (!new_child) {
//     //     new_child = object->requireUniqueObjectInThisCoordinates(getChild(), path);
//     //     new_child->translation = - new_child->translation;
//     // }
//     // if (copyEdges) {
//     //     std::unique_ptr<Geometry2DCartesian> result(new Geometry2DCartesian(*this));
//     //     result->extrusion = plask::make_shared<Extrusion>(new_child, getExtrusion()->length);
//     //     return result.release();
//     // } else
//     //     return new Geometry2DCartesian(new_child, getExtrusion()->length);
// }

void Geometry2DCartesian::setEdges(Direction direction, const edge::Strategy& border_lo, const edge::Strategy& border_hi) {
    Primitive<3>::ensureIsValid2DDirection(direction);
    if (direction == DIRECTION_TRAN)
        leftright.setStrategies(border_lo, border_hi);
    else
        bottomup.setStrategies(border_lo, border_hi);
    fireChanged(Event::EVENT_EDGES);
}

void Geometry2DCartesian::setEdge(Direction direction, bool higher, const edge::Strategy& border_to_set) {
    Primitive<3>::ensureIsValid2DDirection(direction);
    if (direction == DIRECTION_TRAN)
        leftright.set(higher, border_to_set);
    else
        bottomup.set(higher, border_to_set);
    fireChanged(Event::EVENT_EDGES);
}

const edge::Strategy& Geometry2DCartesian::getEdge(Direction direction, bool higher) const {
    Primitive<3>::ensureIsValid2DDirection(direction);
    return (direction == DIRECTION_TRAN) ? leftright.get(higher) : bottomup.get(higher);
}

shared_ptr<GeometryObject> Geometry2DCartesian::shallowCopy() const {
    shared_ptr<Geometry2DCartesian> result = make_shared<Geometry2DCartesian>(static_pointer_cast<Extrusion>(this->extrusion->shallowCopy()));
    result->setEdges(DIRECTION_TRAN, leftright.getLo(), leftright.getHi());
    result->setEdges(DIRECTION_VERT, bottomup.getLo(), bottomup.getHi());
    result->frontMaterial = frontMaterial;
    result->backMaterial = backMaterial;
    return result;
}

shared_ptr<GeometryObject> Geometry2DCartesian::deepCopy(std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const {
    auto found = copied.find(this);
    if (found != copied.end()) return found->second;
    shared_ptr<Geometry2DCartesian> result = make_shared<Geometry2DCartesian>(static_pointer_cast<Extrusion>(this->extrusion->deepCopy(copied)));
    result->setEdges(DIRECTION_TRAN, leftright.getLo(), leftright.getHi());
    result->setEdges(DIRECTION_VERT, bottomup.getLo(), bottomup.getHi());
    result->frontMaterial = frontMaterial;
    result->backMaterial = backMaterial;
    copied[this] = result;
    return result;
}

void Geometry2DCartesian::writeXML(XMLWriter::Element& parent_xml_object, WriteXMLCallback& write_cb, AxisNames axes) const {
    XMLWriter::Element tag = write_cb.makeTag(parent_xml_object, *this, axes);
    if (WriteXMLCallback::isRef(tag)) return;
    writeXMLAttr(tag, axes);
    if (auto c = getExtrusion()) {
        if (isinf(c->getLength()) && c->hasChild()) {
            c->getChild()->writeXML(tag, write_cb, axes);
        } else {
            c->writeXML(tag, write_cb, axes);
        }
    }
}

Geometry2DCylindrical::Geometry2DCylindrical(shared_ptr<Revolution> revolution)
    : revolution(revolution)
{
    initNewChild();
}

Geometry2DCylindrical::Geometry2DCylindrical(shared_ptr<GeometryObjectD<2>> childGeometry)
    : revolution(plask::make_shared<Revolution>(childGeometry))
{
   initNewChild();
}

shared_ptr< GeometryObjectD<2> > Geometry2DCylindrical::getChild() const {
    if (!revolution) throw NoChildException();
    auto child = revolution->getChild();
    if (!child) throw NoChildException();
    return child;
}

shared_ptr< GeometryObjectD<2> > Geometry2DCylindrical::getChildUnsafe() const {
    return revolution->getChild();
}

shared_ptr<Material> Geometry2DCylindrical::getMaterial(const Vec<2, double> &p) const {
    Vec<2, double> r = p;
    if (r.c0 < 0) r.c0 = -r.c0;  // Structure is ALWAYS symmetric with respect to the axis

    shared_ptr<Material> material;

    bottomup.apply(cachedBoundingBox, r, material);
    if (material) return material;

    innerouter.apply(cachedBoundingBox, r, material);
    if (material) return material;

    return getMaterialOrDefault(r);
}

void Geometry2DCylindrical::setRevolution(shared_ptr<Revolution> revolution) {
    if (this->revolution == revolution) return;
    this->revolution = revolution;
    this->initNewChild();
    fireChildrenChanged();
}

// Geometry2DCylindrical* Geometry2DCylindrical::getSubspace(const shared_ptr< GeometryObjectD<2> >& object, const PathHints* path, bool copyEdges) const {
// }

void Geometry2DCylindrical::setEdges(Direction direction, const edge::Strategy& border_to_set) {
    Primitive<3>::ensureIsValid2DDirection(direction);
    if (direction == DIRECTION_TRAN) {
        try {
            innerouter.setBoth(dynamic_cast<const edge::UniversalStrategy&>(border_to_set));
        } catch (std::bad_cast) {
            throw BadInput("setEdges", "Wrong edge type for inner or outer edge");
        }
    } else
        bottomup.setBoth(border_to_set);
    fireChanged(Event::EVENT_EDGES);
}

void Geometry2DCylindrical::setEdges(Direction direction, const edge::Strategy& border_lo, const edge::Strategy& border_hi) {
    ensureBoundDirIsProper(direction/*, false*/);
    //ensureBoundDirIsProper(direction, true);
    bottomup.setStrategies(border_lo, border_hi);   //bottomup is only one valid proper bound for lo and hi
    fireChanged(Event::EVENT_EDGES);
}

void Geometry2DCylindrical::setEdge(Direction direction, bool higher, const edge::Strategy& border_to_set) {
    ensureBoundDirIsProper(direction/*, higher*/);
    if (direction == DIRECTION_TRAN) {
        try {
            innerouter.set(higher, dynamic_cast<const edge::UniversalStrategy&>(border_to_set));
        } catch (std::bad_cast) {
            throw BadInput("setEdge", "Wrong edge type for inner or outer edge");
        }
    } else
        bottomup.set(higher, border_to_set);
    fireChanged(Event::EVENT_EDGES);
}

const edge::Strategy& Geometry2DCylindrical::getEdge(Direction direction, bool higher) const {
    ensureBoundDirIsProper(direction/*, higher*/);
    return (direction == DIRECTION_TRAN) ? innerouter.get(higher) : bottomup.get(higher);
}

shared_ptr<GeometryObject> Geometry2DCylindrical::shallowCopy() const {
    shared_ptr<Geometry2DCylindrical> result = make_shared<Geometry2DCylindrical>(static_pointer_cast<Revolution>(static_pointer_cast<Revolution>(this->revolution->shallowCopy())));
    result->setEdges(DIRECTION_TRAN, innerouter.getLo(), innerouter.getHi());
    result->setEdges(DIRECTION_VERT, bottomup.getLo(), bottomup.getHi());
    return result;
}

shared_ptr<GeometryObject> Geometry2DCylindrical::deepCopy(std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const {
    auto found = copied.find(this);
    if (found != copied.end()) return found->second;
    shared_ptr<Geometry2DCylindrical> result = make_shared<Geometry2DCylindrical>(static_pointer_cast<Revolution>(this->revolution->deepCopy(copied)));
    result->setEdges(DIRECTION_TRAN, innerouter.getLo(), innerouter.getHi());
    result->setEdges(DIRECTION_VERT, bottomup.getLo(), bottomup.getHi());
    copied[this] = result;
    return result;
}

void Geometry2DCylindrical::writeXML(XMLWriter::Element& parent_xml_object, WriteXMLCallback& write_cb, AxisNames axes) const {
    XMLWriter::Element tag = write_cb.makeTag(parent_xml_object, *this, axes);
    if (WriteXMLCallback::isRef(tag)) return;
    writeXMLAttr(tag, axes);
    if (auto c = getRevolution()) c->writeXML(tag, write_cb, axes);
}

void Geometry3D::setEdges(Direction direction, const edge::Strategy &border_lo, const edge::Strategy &border_hi) {
    switch (direction) {
        case DIRECTION_LONG: backfront.setStrategies(border_lo, border_hi); break;
        case DIRECTION_TRAN: leftright.setStrategies(border_lo, border_hi); break;
        case DIRECTION_VERT: bottomup.setStrategies(border_lo, border_hi); break;
    }
    fireChanged(Event::EVENT_EDGES);
}

void Geometry3D::setEdges(Direction direction, const edge::Strategy &border_to_set) {
    switch (direction) {
        case DIRECTION_LONG: backfront.setBoth(border_to_set); break;
        case DIRECTION_TRAN: leftright.setBoth(border_to_set); break;
        case DIRECTION_VERT: bottomup.setBoth(border_to_set); break;
    }
    fireChanged(Event::EVENT_EDGES);
}

void Geometry3D::setEdge(Direction direction, bool higher, const edge::Strategy &border_to_set) {
    switch (direction) {
        case DIRECTION_LONG: backfront.set(higher, border_to_set); break;
        case DIRECTION_TRAN: leftright.set(higher, border_to_set); break;
        case DIRECTION_VERT: bottomup.set(higher, border_to_set); break;
    }
    fireChanged(Event::EVENT_EDGES);
}

const edge::Strategy &Geometry3D::getEdge(Direction direction, bool higher) const {
    switch (direction) {
        case DIRECTION_LONG: return backfront.get(higher);
        case DIRECTION_TRAN: return leftright.get(higher);
        case DIRECTION_VERT: return bottomup.get(higher);
    }
    assert(0);
#ifdef _MSC_VER
	__assume(0);
#endif
    std::abort();   // to silent warning in gcc/clang release build
}

Geometry3D::Geometry3D(shared_ptr<GeometryObjectD<3> > child)
: child(child) {
    initNewChild();
}

shared_ptr<GeometryObjectD<3> > Geometry3D::getChild() const {
    if (!child) throw NoChildException();
    return child;
}

shared_ptr< GeometryObjectD<3> > Geometry3D::getChildUnsafe() const {
    return child;
}

shared_ptr<GeometryObjectD<3> > Geometry3D::getObject3D() const {
    return child;
}

shared_ptr<Material> Geometry3D::getMaterial(const Vec<3, double> &p) const {
    Vec<3, double> r = p;
    shared_ptr<Material> material;

    bottomup.apply(cachedBoundingBox, r, material);
    if (material) return material;

    leftright.apply(cachedBoundingBox, r, material);
    if (material) return material;

    backfront.apply(cachedBoundingBox, r, material);
    if (material) return material;

    return getMaterialOrDefault(r);
}

shared_ptr<GeometryObject> Geometry3D::shallowCopy() const {
    shared_ptr<Geometry3D> result = make_shared<Geometry3D>(this->child);
    result->setEdges(DIRECTION_LONG, backfront.getLo(), backfront.getHi());
    result->setEdges(DIRECTION_TRAN, leftright.getLo(), leftright.getHi());
    result->setEdges(DIRECTION_VERT, bottomup.getLo(), bottomup.getHi());
    return result;
}

shared_ptr<GeometryObject> Geometry3D::deepCopy(std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const {
    auto found = copied.find(this);
    if (found != copied.end()) return found->second;
    shared_ptr<Geometry3D> result = make_shared<Geometry3D>(static_pointer_cast<GeometryObjectD<3>>(this->child->deepCopy(copied)));
    result->setEdges(DIRECTION_LONG, backfront.getLo(), backfront.getHi());
    result->setEdges(DIRECTION_TRAN, leftright.getLo(), leftright.getHi());
    result->setEdges(DIRECTION_VERT, bottomup.getLo(), bottomup.getHi());
    copied[this] = result;
    return result;
}

// Geometry3D* Geometry3D::getSubspace(const shared_ptr<GeometryObjectD<3>>& object, const PathHints* path, bool copyEdges) const {
// }



}   // namespace plask
