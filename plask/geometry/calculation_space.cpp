#include "calculation_space.h"

#include <memory>

namespace plask {

Space2dCartesian::Space2dCartesian(const shared_ptr<Extrusion>& extrusion)
    : extrusion(extrusion)
{
    init();
}

Space2dCartesian::Space2dCartesian(const shared_ptr<GeometryElementD<2>>& childGeometry, double length)
    : extrusion(make_shared<Extrusion>(childGeometry, length))
{
   init();
}

shared_ptr< GeometryElementD<2> > Space2dCartesian::getChild() const {
    return extrusion->getChild();
}

shared_ptr<Material> Space2dCartesian::getMaterial(const Vec<2, double> &p) const {
    Vec<2, double> r = p;
    shared_ptr<Material> material;

    bottomup.apply(cachedBoundingBox, r, material);
    if (material) return material;

    leftright.apply(cachedBoundingBox, r, material);
    if (material) return material;

    return getMaterialOrDefault(r);
}

Space2dCartesian* Space2dCartesian::getSubspace(const shared_ptr< GeometryElementD<2> >& element, const PathHints* path, bool copyBorders) const {
    auto new_child = getChild()->getElementInThisCordinates(element, path);
    if (copyBorders) {
        std::unique_ptr<Space2dCartesian> result(new Space2dCartesian(*this));
        result->extrusion = make_shared<Extrusion>(new_child, getExtrusion()->length);
        return result.release();
    } else    
        return new Space2dCartesian(new_child, getExtrusion()->length);
    
}

void Space2dCartesian::setBorders(Primitive<3>::DIRECTION direction, const border::Strategy& border_to_set) {
    Primitive<3>::ensureIsValid2dDirection(direction);
    if (direction == Primitive<3>::DIRECTION_TRAN)
        leftright.setBoth(border_to_set);
    else
        bottomup.setBoth(border_to_set);
}

void Space2dCartesian::setBorder(Primitive<3>::DIRECTION direction, bool higher, const border::Strategy& border_to_set) {
    Primitive<3>::ensureIsValid2dDirection(direction);
    if (direction == Primitive<3>::DIRECTION_TRAN)
        leftright.set(higher, border_to_set);
    else
        bottomup.set(higher, border_to_set);
}

const border::Strategy& Space2dCartesian::getBorder(Primitive<3>::DIRECTION direction, bool higher) const {
    Primitive<3>::ensureIsValid2dDirection(direction);
    return (direction == Primitive<3>::DIRECTION_TRAN) ? leftright.get(higher) : bottomup.get(higher);
}

Space2dCylindrical::Space2dCylindrical(const shared_ptr<Revolution>& revolution)
    : revolution(revolution)
{
    init();
}

Space2dCylindrical::Space2dCylindrical(const shared_ptr<GeometryElementD<2>>& childGeometry)
    : revolution(make_shared<Revolution>(childGeometry))
{
   init();
}

shared_ptr< GeometryElementD<2> > Space2dCylindrical::getChild() const {
    return revolution->getChild();
}

shared_ptr<Material> Space2dCylindrical::getMaterial(const Vec<2, double> &p) const {
    Vec<2, double> r = p;
    shared_ptr<Material> material;

    bottomup.apply(cachedBoundingBox, r, material);
    if (material) return material;

    outer.applyIfHi(cachedBoundingBox, r, material);
    if (material) return material;

    return getMaterialOrDefault(r);
}

Space2dCylindrical* Space2dCylindrical::getSubspace(const shared_ptr< GeometryElementD<2> >& element, const PathHints* path, bool copyBorders) const {
    auto new_child = getChild()->getElementInThisCordinates(element, path);
    if (copyBorders) {
        std::unique_ptr<Space2dCylindrical> result(new Space2dCylindrical(*this));
        result->revolution = make_shared<Revolution>(new_child);
        return result.release();
    } else    
        return new Space2dCylindrical(new_child);
}

void Space2dCylindrical::setBorders(Primitive<3>::DIRECTION direction, const border::Strategy& border_to_set) {
    Primitive<3>::ensureIsValid2dDirection(direction);
    if (direction == Primitive<3>::DIRECTION_TRAN)
        outer = castBorder<border::UniversalStrategy>(border_to_set);
    else
        bottomup.setBoth(border_to_set);
}

void Space2dCylindrical::setBorder(Primitive<3>::DIRECTION direction, bool higher, const border::Strategy& border_to_set) {
    ensureBoundDirIsProper(direction, higher);
    if (direction == Primitive<3>::DIRECTION_TRAN) {
        outer = castBorder<border::UniversalStrategy>(border_to_set);
    } else
        bottomup.set(higher, border_to_set);
}

const border::Strategy& Space2dCylindrical::getBorder(Primitive<3>::DIRECTION direction, bool higher) const {
    ensureBoundDirIsProper(direction, higher);
    return (direction == Primitive<3>::DIRECTION_TRAN) ? outer.getStrategy() : bottomup.get(higher);
}

}   // namespace plask
