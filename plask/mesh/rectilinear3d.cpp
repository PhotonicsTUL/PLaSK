#include "rectilinear3d.h"

namespace plask {

//TODO I think that this method should split space-changers
//(does getLeafsBoundingBoxes already do it?)
void RectilinearMesh3d::buildFromGeometry(const GeometryElementD<3>& geometry) {
    std::vector<Box3d> boxes = geometry.getLeafsBoundingBoxes();

    for (auto box: boxes) {
        c0.addPoint(box.lower.c0);
        c0.addPoint(box.upper.c0);
        c1.addPoint(box.lower.c1);
        c1.addPoint(box.upper.c1);
        c2.addPoint(box.lower.c2);
        c2.addPoint(box.upper.c2);
    }
}

} // namespace plask
