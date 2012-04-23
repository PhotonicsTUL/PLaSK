#include "calculation_space.h"

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

}   // namespace plask
