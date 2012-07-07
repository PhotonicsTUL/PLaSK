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


} // namespace plask