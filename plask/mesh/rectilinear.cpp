#include "rectilinear.h"

#include "rectangular2d_impl.h"
#include "rectangular3d_impl.h"

namespace plask {

template class RectangularMesh2d<RectilinearMesh1d>;
template class RectangularMesh3d<RectilinearMesh1d>;

void RectilinearMesh2d::buildFromGeometry(const GeometryElementD<2>& geometry) {
    std::vector<Box2d> boxes = geometry.getLeafsBoundingBoxes();

    for (auto& box: boxes) {
        c0.addPoint(box.lower.c0);
        c0.addPoint(box.upper.c0);
        c1.addPoint(box.lower.c1);
        c1.addPoint(box.upper.c1);
    }

    fireChanged();
}

RectilinearMesh2d RectilinearMesh2d::getMidpointsMesh() const {

    if (c0.size() < 2 || c1.size() < 2) throw BadMesh("getMidpointsMesh", "at least two points in each direction are required");

    RectilinearMesh1d line0;
    for (auto a = c0.begin(), b = c0.begin()+1; b != c0.end(); ++a, ++b)
        line0.addPoint(0.5 * (*a + *b));

    RectilinearMesh1d line1;
    for (auto a = c1.begin(), b = c1.begin()+1; b != c1.end(); ++a, ++b)
        line1.addPoint(0.5 * (*a + *b));

    return RectilinearMesh2d(line0, line1, getIterationOrder());
}

void RectilinearMesh3d::buildFromGeometry(const GeometryElementD<3>& geometry) {
    std::vector<Box3d> boxes = geometry.getLeafsBoundingBoxes();

    for (auto& box: boxes) {
        c0.addPoint(box.lower.c0);
        c0.addPoint(box.upper.c0);
        c1.addPoint(box.lower.c1);
        c1.addPoint(box.upper.c1);
        c2.addPoint(box.lower.c2);
        c2.addPoint(box.upper.c2);
    }

    fireChanged();
}


RectilinearMesh3d RectilinearMesh3d::getMidpointsMesh() const {

    if (c0.size() < 2 || c1.size() < 2 || c2.size() < 2) throw BadMesh("getMidpointsMesh", "at least two points in each direction are required");

    RectilinearMesh1d line0;
    for (auto a = c0.begin(), b = c0.begin()+1; b != c0.end(); ++a, ++b)
        line0.addPoint(0.5 * (*a + *b));

    RectilinearMesh1d line1;
    for (auto a = c1.begin(), b = c1.begin()+1; b != c1.end(); ++a, ++b)
        line1.addPoint(0.5 * (*a + *b));

    RectilinearMesh1d line2;
    for (auto a = c2.begin(), b = c2.begin()+1; b != c2.end(); ++a, ++b)
        line2.addPoint(0.5 * (*a + *b));

    return RectilinearMesh3d(line0, line1, line2, getIterationOrder());
}

} // namespace plask