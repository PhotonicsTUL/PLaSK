#include "calculation_space.h"

namespace plask {

shared_ptr< GeometryElementD<2> > Space2DCartesian::getChild() const {
    return extrusion->getChild();
}

void Space2DCartesianDragToEdge::onChildChanged(const GeometryElement::Event &evt) {
    if (evt.isResize())
        cachedBoundingBox = getChild()->getBoundingBox();
}

Space2DCartesianDragToEdge::Space2DCartesianDragToEdge(const shared_ptr<Extrusion> &extrusion, Space2DCartesianDragToEdge::ExtendType extendType)
    : Space2DCartesian(extrusion), _extend(extendType), cachedBoundingBox(extrusion->getChild()->getBoundingBox())
{
    extrusion->getChild()->changedConnectMethod(this, &Space2DCartesianDragToEdge::onChildChanged);
}

shared_ptr<Material> Space2DCartesianDragToEdge::getMaterial(const Vec<2, double> &p) const {
    Vec<2, double> r = p;
    if (_extend & EXTEND_TRAN) {
        if (r.tran > cachedBoundingBox.upper.tran) r.tran = cachedBoundingBox.upper.tran;
        else if (r.tran < cachedBoundingBox.lower.tran) r.tran = cachedBoundingBox.lower.tran;
    }
    if (_extend & EXTEND_VERTICAL) {
        if (r.up > cachedBoundingBox.upper.up) r.up = cachedBoundingBox.upper.up;
        else if (r.up < cachedBoundingBox.lower.up) r.up = cachedBoundingBox.lower.up;
    }
    return getMaterialOrDefault(r);
}

}   // namespace plask
