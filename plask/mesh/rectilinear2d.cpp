#include "rectilinear2d.h"

namespace plask {

void RectilinearMesh2d::buildFromGeometry(const GeometryElementD<2>& geometry) {
    std::vector<Box2d> boxes = geometry.getLeafsBoundingBoxes();

    for (auto box: boxes) {
        c0.addPoint(box.lower.c0);
        c0.addPoint(box.upper.c0);
        c1.addPoint(box.lower.c1);
        c1.addPoint(box.upper.c1);
    }
}

} // namespace plask
