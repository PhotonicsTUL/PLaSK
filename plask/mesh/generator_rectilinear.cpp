#include <deque>

#include <plask/log/log.h>
#include "generator_rectilinear.h"

namespace plask {

RectilinearMesh1D RectilinearMesh2DSimpleGenerator::get1DMesh(const RectilinearMesh1D& initial, const shared_ptr<GeometryElementD<2>>& geometry, size_t dir)
{
    // TODO: Użyj algorytmu Roberta, może będzie lepszy

    RectilinearMesh1D result = initial;

    // First add refinement points
    for (auto ref: refinements[dir]) {
        auto boxes = geometry->getLeafsBoundingBoxes(&ref.first);
        if (boxes.size() > 1) writelog(LOG_WARNING, "RectilinearMesh2DSimpleGenerator: Single refinement defined for more than one object.");
        for (auto box: boxes) {
            for (auto x: ref.second) {
                if (x < 0 || x > box.upper[dir]-box.lower[dir])
                    writelog(LOG_WARNING, "RectilinearMesh2DSimpleGenerator: Refinement at %1% outside of the object (0 ... %2%).", x, box.upper[dir]-box.lower[dir]);
                result.addPoint(box.lower[dir] + x);
            }
        }
    }

    // First divide each element
    double x = *result.begin();
    std::vector<double> points; points.reserve((division-1)*(result.size()-1));
    for (auto i = result.begin()+1; i!= result.end(); ++i) {
        double w = *i - x;
        for (size_t j = 1; j != division; ++j) points.push_back(x + w*j/division);
        x = *i;
    }
    result.addOrderedPoints(points.begin(), points.end());

    if (result.size() < 3) return result;

    // Now ensure, that the grids do not change to quickly
    bool repeat;
    do {
        double w_prev = INFINITY, w = result[1]-result[0], w_next = result[2]-result[1];
        repeat = false;
        for (auto i = result.begin()+1; i != result.end(); ++i) {
            if (w > 2.*w_prev || w > 2.*w_next) {
                result.addPoint(0.5 * (*(i-1) + *i));
                repeat = true;
                break;
            }
            w_prev = w;
            w = w_next;
            w_next = (i+2 == result.end())? INFINITY : *(i+2) - *(i+1);
        }
    } while (repeat);
    return result;
}

shared_ptr<RectilinearMesh2D> RectilinearMesh2DSimpleGenerator::generate(const shared_ptr<GeometryElementD<2>>& geometry)
{
    RectilinearMesh2D initial = RectilinearMeshFromGeometry(geometry);
    auto mesh = make_shared<RectilinearMesh2D>();
    mesh->c0 = get1DMesh(initial.c0, geometry, 0);
    mesh->c1 = get1DMesh(initial.c1, geometry, 1);
    return mesh;
}

} // namespace plask