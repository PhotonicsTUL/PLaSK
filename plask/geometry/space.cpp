#include "space.h"

#include <memory>

namespace plask {

const std::map<std::string, std::string> Geometry::null_borders;

void Geometry::setBorders(const std::function<boost::optional<std::string>(const std::string& s)>& borderValuesGetter, const AxisNames& axesNames)
{
    boost::optional<std::string> v, v_lo, v_hi;
    v = borderValuesGetter("borders");
    if (v) setAllBorders(*border::Strategy::fromStrUnique(*v));
    v = borderValuesGetter("planar");
    if (v) setPlanarBorders(*border::Strategy::fromStrUnique(*v));
    for (int dir_nr = 0; dir_nr < 3; ++dir_nr) {
        std::string axis_name = axesNames[dir_nr];
        v = borderValuesGetter(axis_name);
        if (v) setBorders(DIRECTION(dir_nr), *border::Strategy::fromStrUnique(*v));
        v_lo = borderValuesGetter(axis_name + "-lo");
        if (v = borderValuesGetter(alternativeDirectionName(dir_nr, 0))) {
            if (v_lo) throw BadInput("setBorders", "Border specified by both '%1%-lo' and '%2%'", axis_name, alternativeDirectionName(dir_nr, 0));
            else v_lo = v;
        }
        v_hi = borderValuesGetter(axis_name + "-hi");
        if (v = borderValuesGetter(alternativeDirectionName(dir_nr, 1))) {
            if (v_hi) throw BadInput("setBorders", "Border specified by both '%1%-hi' and '%2%'", axis_name, alternativeDirectionName(dir_nr, 1));
            else v_hi = v;
        }
        try {
            if (v_lo && v_hi) {
                setBorders(DIRECTION(dir_nr),  *border::Strategy::fromStrUnique(*v_lo), *border::Strategy::fromStrUnique(*v_hi));
            } else {
                if (v_lo) setBorder(DIRECTION(dir_nr), false, *border::Strategy::fromStrUnique(*v_lo));
                if (v_hi) setBorder(DIRECTION(dir_nr), true, *border::Strategy::fromStrUnique(*v_hi));
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
    setBorders(DIRECTION_LON, border_to_set);
    setBorders(DIRECTION_TRAN, border_to_set);
}

Geometry2DCartesian::Geometry2DCartesian(const shared_ptr<Extrusion>& extrusion)
    : extrusion(extrusion)
{
    init();
}

Geometry2DCartesian::Geometry2DCartesian(const shared_ptr<GeometryElementD<2>>& childGeometry, double length)
    : extrusion(make_shared<Extrusion>(childGeometry, length))
{
   init();
}

shared_ptr< GeometryElementD<2> > Geometry2DCartesian::getChild() const {
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

Geometry2DCartesian* Geometry2DCartesian::getSubspace(const shared_ptr<GeometryElementD<2>>& element, const PathHints* path, bool copyBorders) const {
    auto new_child = getChild()->getElementInThisCordinates(element, path);
    if (!new_child) {
        new_child = element->requireElementInThisCordinates(getChild(), path);
        new_child->translation = - new_child->translation;
    }
    if (copyBorders) {
        std::unique_ptr<Geometry2DCartesian> result(new Geometry2DCartesian(*this));
        result->extrusion = make_shared<Extrusion>(new_child, getExtrusion()->length);
        return result.release();
    } else
        return new Geometry2DCartesian(new_child, getExtrusion()->length);
}

void Geometry2DCartesian::setBorders(DIRECTION direction, const border::Strategy& border_lo, const border::Strategy& border_hi) {
    Primitive<3>::ensureIsValid2DDirection(direction);
    if (direction == DIRECTION_TRAN)
        leftright.setStrategies(border_lo, border_hi);
    else
        bottomup.setStrategies(border_lo, border_hi);
    fireChanged(Event::BORDERS);
}

void Geometry2DCartesian::setBorder(DIRECTION direction, bool higher, const border::Strategy& border_to_set) {
    Primitive<3>::ensureIsValid2DDirection(direction);
    if (direction == DIRECTION_TRAN)
        leftright.set(higher, border_to_set);
    else
        bottomup.set(higher, border_to_set);
    fireChanged(Event::BORDERS);
}

const border::Strategy& Geometry2DCartesian::getBorder(DIRECTION direction, bool higher) const {
    Primitive<3>::ensureIsValid2DDirection(direction);
    return (direction == DIRECTION_TRAN) ? leftright.get(higher) : bottomup.get(higher);
}

Geometry2DCylindrical::Geometry2DCylindrical(const shared_ptr<Revolution>& revolution)
    : revolution(revolution)
{
    init();
}

Geometry2DCylindrical::Geometry2DCylindrical(const shared_ptr<GeometryElementD<2>>& childGeometry)
    : revolution(make_shared<Revolution>(childGeometry))
{
   init();
}

shared_ptr< GeometryElementD<2> > Geometry2DCylindrical::getChild() const {
    return revolution->getChild();
}

shared_ptr<Material> Geometry2DCylindrical::getMaterial(const Vec<2, double> &p) const {
    Vec<2, double> r = p;
    shared_ptr<Material> material;

    bottomup.apply(cachedBoundingBox, r, material);
    if (material) return material;

    outer.applyIfHi(cachedBoundingBox, r, material);
    if (material) return material;

    return getMaterialOrDefault(r);
}

Geometry2DCylindrical* Geometry2DCylindrical::getSubspace(const shared_ptr< GeometryElementD<2> >& element, const PathHints* path, bool copyBorders) const {
    auto new_child = getChild()->getElementInThisCordinates(element, path);
    if (!new_child) {
        new_child = element->requireElementInThisCordinates(getChild(), path);
        new_child->translation = - new_child->translation;
    }
    if (copyBorders) {
        std::unique_ptr<Geometry2DCylindrical> result(new Geometry2DCylindrical(*this));
        result->revolution = make_shared<Revolution>(new_child);
        return result.release();
    } else
        return new Geometry2DCylindrical(new_child);
}

void Geometry2DCylindrical::setBorders(DIRECTION direction, const border::Strategy& border_to_set) {
    Primitive<3>::ensureIsValid2DDirection(direction);
    if (direction == DIRECTION_TRAN)
        outer = castBorder<border::UniversalStrategy>(border_to_set);
    else
        bottomup.setBoth(border_to_set);
    fireChanged(Event::BORDERS);
}

void Geometry2DCylindrical::setBorders(DIRECTION direction, const border::Strategy& border_lo, const border::Strategy& border_hi) {
    ensureBoundDirIsProper(direction, false);
    ensureBoundDirIsProper(direction, true);
    bottomup.setStrategies(border_lo, border_hi);   //bottomup is only one valid proper bound for lo and hi
    fireChanged(Event::BORDERS);
}

void Geometry2DCylindrical::setBorder(DIRECTION direction, bool higher, const border::Strategy& border_to_set) {
    ensureBoundDirIsProper(direction, higher);
    if (direction == DIRECTION_TRAN) {
        outer = castBorder<border::UniversalStrategy>(border_to_set);
    } else
        bottomup.set(higher, border_to_set);
    fireChanged(Event::BORDERS);
}

const border::Strategy& Geometry2DCylindrical::getBorder(DIRECTION direction, bool higher) const {
    ensureBoundDirIsProper(direction, higher);
    return (direction == DIRECTION_TRAN) ? outer.getStrategy() : bottomup.get(higher);
}

}   // namespace plask
