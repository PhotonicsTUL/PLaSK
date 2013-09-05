#include "space.h"

#include <memory>
#include <cassert>

namespace plask {

const std::map<std::string, std::string> Geometry::null_borders;

void Geometry::setBorders(const std::function<boost::optional<std::string>(const std::string& s)>& borderValuesGetter, const AxisNames& axesNames, const MaterialsSource &materialsSource)
{
    boost::optional<std::string> v, v_lo, v_hi;
    v = borderValuesGetter("borders");
    if (v) setAllBorders(*border::Strategy::fromStrUnique(*v, materialsSource));
    v = borderValuesGetter("planar");
    if (v) setPlanarBorders(*border::Strategy::fromStrUnique(*v, materialsSource));
    for (int dir_nr = 0; dir_nr < 3; ++dir_nr) {
        std::string axis_name = axesNames[dir_nr];
        v = borderValuesGetter(axis_name);
        if (v) setBorders(Direction(dir_nr), *border::Strategy::fromStrUnique(*v, materialsSource));
        v_lo = borderValuesGetter(axis_name + "-lo");
        if ((v = borderValuesGetter(alternativeDirectionName(dir_nr, 0)))) {
            if (v_lo) throw BadInput("setBorders", "Border specified by both '%1%-lo' and '%2%'", axis_name, alternativeDirectionName(dir_nr, 0));
            else v_lo = v;
        }
        v_hi = borderValuesGetter(axis_name + "-hi");
        if ((v = borderValuesGetter(alternativeDirectionName(dir_nr, 1)))) {
            if (v_hi) throw BadInput("setBorders", "Border specified by both '%1%-hi' and '%2%'", axis_name, alternativeDirectionName(dir_nr, 1));
            else v_hi = v;
        }
        try {
            if (v_lo && v_hi) {
                setBorders(Direction(dir_nr),  *border::Strategy::fromStrUnique(*v_lo, materialsSource), *border::Strategy::fromStrUnique(*v_hi, materialsSource));
            } else {
                if (v_lo) setBorder(Direction(dir_nr), false, *border::Strategy::fromStrUnique(*v_lo, materialsSource));
                if (v_hi) setBorder(Direction(dir_nr), true, *border::Strategy::fromStrUnique(*v_hi, materialsSource));
            }
        } catch (DimensionError) {
            throw BadInput("setBorders", "Axis '%1%' is not allowed for this space", axis_name);
        }
    }
}


template <>
void GeometryD<2>::setPlanarBorders(const border::Strategy& border_to_set) {
    setBorders(DIRECTION_TRAN, border_to_set);
}

template <>
void GeometryD<3>::setPlanarBorders(const border::Strategy& border_to_set) {
    setBorders(DIRECTION_LONG, border_to_set);
    setBorders(DIRECTION_TRAN, border_to_set);
}

Geometry2DCartesian::Geometry2DCartesian(shared_ptr<Extrusion> extrusion)
    : extrusion(extrusion)
{
    initNewChild();
}

Geometry2DCartesian::Geometry2DCartesian(shared_ptr<GeometryObjectD<2>> childGeometry, double length)
    : extrusion(make_shared<Extrusion>(childGeometry, length))
{
   initNewChild();
}

shared_ptr< GeometryObjectD<2> > Geometry2DCartesian::getChild() const {
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

// Geometry2DCartesian* Geometry2DCartesian::getSubspace(const shared_ptr<GeometryObjectD<2>>& object, const PathHints* path, bool copyBorders) const {
//     auto shifts = getChild()->getObjectPositions(object, path);
//     // auto new_child = getChild()->getUniqueObjectInThisCoordinates(object, path);
//     // if (!new_child) {
//     //     new_child = object->requireUniqueObjectInThisCoordinates(getChild(), path);
//     //     new_child->translation = - new_child->translation;
//     // }
//     // if (copyBorders) {
//     //     std::unique_ptr<Geometry2DCartesian> result(new Geometry2DCartesian(*this));
//     //     result->extrusion = make_shared<Extrusion>(new_child, getExtrusion()->length);
//     //     return result.release();
//     // } else
//     //     return new Geometry2DCartesian(new_child, getExtrusion()->length);
// }

void Geometry2DCartesian::setBorders(Direction direction, const border::Strategy& border_lo, const border::Strategy& border_hi) {
    Primitive<3>::ensureIsValid2DDirection(direction);
    if (direction == DIRECTION_TRAN)
        leftright.setStrategies(border_lo, border_hi);
    else
        bottomup.setStrategies(border_lo, border_hi);
    fireChanged(Event::EVENT_BORDERS);
}

void Geometry2DCartesian::setBorder(Direction direction, bool higher, const border::Strategy& border_to_set) {
    Primitive<3>::ensureIsValid2DDirection(direction);
    if (direction == DIRECTION_TRAN)
        leftright.set(higher, border_to_set);
    else
        bottomup.set(higher, border_to_set);
    fireChanged(Event::EVENT_BORDERS);
}

const border::Strategy& Geometry2DCartesian::getBorder(Direction direction, bool higher) const {
    Primitive<3>::ensureIsValid2DDirection(direction);
    return (direction == DIRECTION_TRAN) ? leftright.get(higher) : bottomup.get(higher);
}

void Geometry2DCartesian::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    //TODO borders
}

void Geometry2DCartesian::writeXML(XMLWriter::Element& parent_xml_object, WriteXMLCallback& write_cb, AxisNames axes) const {
    XMLWriter::Element tag = write_cb.makeTag(parent_xml_object, *this, axes);
    if (WriteXMLCallback::isRef(tag)) return;
    writeXMLAttr(tag, axes);
    if (auto c = getExtrusion()) c->writeXML(tag, write_cb, axes);
}

Geometry2DCylindrical::Geometry2DCylindrical(shared_ptr<Revolution> revolution)
    : revolution(revolution)
{
    initNewChild();
}

Geometry2DCylindrical::Geometry2DCylindrical(shared_ptr<GeometryObjectD<2>> childGeometry)
    : revolution(make_shared<Revolution>(childGeometry))
{
   initNewChild();
}

shared_ptr< GeometryObjectD<2> > Geometry2DCylindrical::getChild() const {
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

// Geometry2DCylindrical* Geometry2DCylindrical::getSubspace(const shared_ptr< GeometryObjectD<2> >& object, const PathHints* path, bool copyBorders) const {
// }

void Geometry2DCylindrical::setBorders(Direction direction, const border::Strategy& border_to_set) {
    Primitive<3>::ensureIsValid2DDirection(direction);
    if (direction == DIRECTION_TRAN) {
        try {
            innerouter.setBoth(dynamic_cast<const border::UniversalStrategy&>(border_to_set));
        } catch (std::bad_cast) {
            throw BadInput("setBorders", "Wrong border type for inner or outer border");
        }
    } else
        bottomup.setBoth(border_to_set);
    fireChanged(Event::EVENT_BORDERS);
}

void Geometry2DCylindrical::setBorders(Direction direction, const border::Strategy& border_lo, const border::Strategy& border_hi) {
    ensureBoundDirIsProper(direction, false);
    ensureBoundDirIsProper(direction, true);
    bottomup.setStrategies(border_lo, border_hi);   //bottomup is only one valid proper bound for lo and hi
    fireChanged(Event::EVENT_BORDERS);
}

void Geometry2DCylindrical::setBorder(Direction direction, bool higher, const border::Strategy& border_to_set) {
    ensureBoundDirIsProper(direction, higher);
    if (direction == DIRECTION_TRAN) {
        try {
            innerouter.set(higher, dynamic_cast<const border::UniversalStrategy&>(border_to_set));
        } catch (std::bad_cast) {
            throw BadInput("setBorder", "Wrong border type for inner or outer border");
        }
    } else
        bottomup.set(higher, border_to_set);
    fireChanged(Event::EVENT_BORDERS);
}

const border::Strategy& Geometry2DCylindrical::getBorder(Direction direction, bool higher) const {
    ensureBoundDirIsProper(direction, higher);
    return (direction == DIRECTION_TRAN) ? innerouter.get(higher) : bottomup.get(higher);
}

void Geometry2DCylindrical::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    //TODO borders
}

void Geometry2DCylindrical::writeXML(XMLWriter::Element& parent_xml_object, WriteXMLCallback& write_cb, AxisNames axes) const {
    XMLWriter::Element tag = write_cb.makeTag(parent_xml_object, *this, axes);
    if (WriteXMLCallback::isRef(tag)) return;
    writeXMLAttr(tag, axes);
    if (auto c = getRevolution()) c->writeXML(tag, write_cb, axes);
}

void Geometry3D::setBorders(Direction direction, const border::Strategy &border_lo, const border::Strategy &border_hi) {
    switch (direction) {
        case DIRECTION_LONG: backfront.setStrategies(border_lo, border_hi); break;
        case DIRECTION_TRAN: leftright.setStrategies(border_lo, border_hi); break;
        case DIRECTION_VERT: bottomup.setStrategies(border_lo, border_hi); break;
    }
    fireChanged(Event::EVENT_BORDERS);
}

void Geometry3D::setBorders(Direction direction, const border::Strategy &border_to_set) {
    switch (direction) {
        case DIRECTION_LONG: backfront.setBoth(border_to_set); break;
        case DIRECTION_TRAN: leftright.setBoth(border_to_set); break;
        case DIRECTION_VERT: bottomup.setBoth(border_to_set); break;
    }
    fireChanged(Event::EVENT_BORDERS);
}

void Geometry3D::setBorder(Direction direction, bool higher, const border::Strategy &border_to_set) {
    switch (direction) {
        case DIRECTION_LONG: backfront.set(higher, border_to_set); break;
        case DIRECTION_TRAN: leftright.set(higher, border_to_set); break;
        case DIRECTION_VERT: bottomup.set(higher, border_to_set); break;
    }
    fireChanged(Event::EVENT_BORDERS);
}

const border::Strategy &Geometry3D::getBorder(Direction direction, bool higher) const {
    switch (direction) {
        case DIRECTION_LONG: return backfront.get(higher);
        case DIRECTION_TRAN: return leftright.get(higher);
        case DIRECTION_VERT: return bottomup.get(higher);
    }
    assert(0);
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

// Geometry3D* Geometry3D::getSubspace(const shared_ptr<GeometryObjectD<3>>& object, const PathHints* path, bool copyBorders) const {
// }

void Geometry3D::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    //TODO borders
}


}   // namespace plask
