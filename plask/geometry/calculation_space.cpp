#include "calculation_space.h"

namespace plask {

Space2DCartesian::Space2DCartesian(const shared_ptr<Extrusion>& extrusion)
    : extrusion(extrusion), cachedBoundingBox(extrusion->getChild()->getBoundingBox())
{
    extrusion->getChild()->changedConnectMethod(this, &Space2DCartesian::onChildChanged);
}

Space2DCartesian::Space2DCartesian(const shared_ptr<GeometryElementD<2>>& childGeometry, double length)
    : extrusion(make_shared<Extrusion>(childGeometry, length)), cachedBoundingBox(childGeometry->getBoundingBox())
{
   childGeometry->changedConnectMethod(this, &Space2DCartesian::onChildChanged);
}

shared_ptr< GeometryElementD<2> > Space2DCartesian::getChild() const {
    return extrusion->getChild();
}

void Space2DCartesian::onChildChanged(const GeometryElement::Event &evt) {
    if (evt.isResize())
        cachedBoundingBox = getChild()->getBoundingBox();
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

}   // namespace plask
