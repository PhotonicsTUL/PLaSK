#include "rectilinear.h"

#include "rectangular2d_impl.h"
#include "rectangular3d_impl.h"

namespace plask {

template class RectangularMesh2D<RectilinearMesh1D>;
template class RectangularMesh3D<RectilinearMesh1D>;

template<>
RectilinearMesh2D RectilinearMesh2D::getMidpointsMesh() const {

    if (c0.size() < 2 || c1.size() < 2) throw BadMesh("getMidpointsMesh", "at least two points in each direction are required");

    RectilinearMesh1D line0;
    for (auto a = c0.begin(), b = c0.begin()+1; b != c0.end(); ++a, ++b)
        line0.addPoint(0.5 * (*a + *b));

    RectilinearMesh1D line1;
    for (auto a = c1.begin(), b = c1.begin()+1; b != c1.end(); ++a, ++b)
        line1.addPoint(0.5 * (*a + *b));

    return RectilinearMesh2D(line0, line1, getIterationOrder());
}

template<>
RectilinearMesh3D RectilinearMesh3D::getMidpointsMesh() const {

    if (c0.size() < 2 || c1.size() < 2 || c2.size() < 2) throw BadMesh("getMidpointsMesh", "at least two points in each direction are required");

    RectilinearMesh1D line0;
    for (auto a = c0.begin(), b = c0.begin()+1; b != c0.end(); ++a, ++b)
        line0.addPoint(0.5 * (*a + *b));

    RectilinearMesh1D line1;
    for (auto a = c1.begin(), b = c1.begin()+1; b != c1.end(); ++a, ++b)
        line1.addPoint(0.5 * (*a + *b));

    RectilinearMesh1D line2;
    for (auto a = c2.begin(), b = c2.begin()+1; b != c2.end(); ++a, ++b)
        line2.addPoint(0.5 * (*a + *b));

    return RectilinearMesh3D(line0, line1, line2, getIterationOrder());
}


RectilinearMesh2D RectilinearMeshFromGeometry(const GeometryElementD<2>& geometry, RectilinearMesh2D::IterationOrder iterationOrder)
{
    RectilinearMesh2D mesh;

    std::vector<Box2D> boxes = geometry.getLeafsBoundingBoxes();

    for (auto& box: boxes) {
        mesh.c0.addPoint(box.lower.c0);
        mesh.c0.addPoint(box.upper.c0);
        mesh.c1.addPoint(box.lower.c1);
        mesh.c1.addPoint(box.upper.c1);
    }

    mesh.setIterationOrder(iterationOrder);

    mesh.fireChanged();

    return mesh;
}

RectilinearMesh3D RectilinearMeshFromGeometry(const GeometryElementD<3>& geometry, RectilinearMesh3D::IterationOrder iterationOrder)
{
    RectilinearMesh3D mesh;

    std::vector<Box3D> boxes = geometry.getLeafsBoundingBoxes();

    for (auto& box: boxes) {
        mesh.c0.addPoint(box.lower.c0);
        mesh.c0.addPoint(box.upper.c0);
        mesh.c1.addPoint(box.lower.c1);
        mesh.c1.addPoint(box.upper.c1);
        mesh.c2.addPoint(box.lower.c2);
        mesh.c2.addPoint(box.upper.c2);
    }

    mesh.setIterationOrder(iterationOrder);

    mesh.fireChanged();

    return mesh;
}

} // namespace plask