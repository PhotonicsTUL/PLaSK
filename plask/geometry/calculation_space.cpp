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

}   // namespace plask
