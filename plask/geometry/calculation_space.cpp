#include "calculation_space.h"

namespace plask {

Space2DCartesian::Space2DCartesian(const shared_ptr<Extrusion>& extrusion)
    : extrusion(extrusion)
{
    init();
}

Space2DCartesian::Space2DCartesian(const shared_ptr<GeometryElementD<2>>& childGeometry, double length)
    : extrusion(make_shared<Extrusion>(childGeometry, length))
{
   init();
}

shared_ptr< GeometryElementD<2> > Space2DCartesian::getChild() const {
    return extrusion->getChild();
}

shared_ptr<Material> Space2DCartesian::getMaterial(const Vec<2, double> &p) const {
    Vec<2, double> r = p;
    shared_ptr<Material> material;

    bottom.applyIfLo(cachedBoundingBox, r, material);
    if (material) return material;
    up.applyIfHi(cachedBoundingBox, r, material);
    if (material) return material;

    left.applyIfLo(cachedBoundingBox, r, material);
    if (material) return material;
    right.applyIfHi(cachedBoundingBox, r, material);
    if (material) return material;

    return getMaterialOrDefault(r);
}


Space2DCylindrical::Space2DCylindrical(const shared_ptr<Revolution>& revolution)
    : revolution(revolution)
{
    init();
}

Space2DCylindrical::Space2DCylindrical(const shared_ptr<GeometryElementD<2>>& childGeometry)
    : revolution(make_shared<Revolution>(childGeometry))
{
   init();
}

shared_ptr< GeometryElementD<2> > Space2DCylindrical::getChild() const {
    return revolution->getChild();
}

shared_ptr<Material> Space2DCylindrical::getMaterial(const Vec<2, double> &p) const {
    Vec<2, double> r = p;
    shared_ptr<Material> material;

    bottom.applyIfLo(cachedBoundingBox, r, material);
    if (material) return material;
    up.applyIfHi(cachedBoundingBox, r, material);
    if (material) return material;

    right.applyIfHi(cachedBoundingBox, r, material);
    if (material) return material;

    return getMaterialOrDefault(r);
}

}   // namespace plask
