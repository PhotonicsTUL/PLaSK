#include <deque>

#include <plask/log/log.h>
#include "generator_rectilinear.h"

namespace plask {

shared_ptr<RectilinearMesh2D> RectilinearMesh2DSimpleGenerator::generate(const shared_ptr<GeometryElementD<2>>& geometry)
{
    auto mesh = make_shared<RectilinearMesh2D>();

    std::vector<Box2D> boxes = geometry->getLeafsBoundingBoxes();

    for (auto& box: boxes) {
        mesh->c0.addPoint(box.lower.c0);
        mesh->c0.addPoint(box.upper.c0);
        mesh->c1.addPoint(box.lower.c1);
        mesh->c1.addPoint(box.upper.c1);
    }

    mesh->setOptimalIterationOrder();
    return mesh;
}

shared_ptr<RectilinearMesh3D> RectilinearMesh3DSimpleGenerator::generate(const shared_ptr<GeometryElementD<3>>& geometry)
{
    auto mesh = make_shared<RectilinearMesh3D>();


    std::vector<Box3D> boxes = geometry->getLeafsBoundingBoxes();

    for (auto& box: boxes) {
        mesh->c0.addPoint(box.lower.c0);
        mesh->c0.addPoint(box.upper.c0);
        mesh->c1.addPoint(box.lower.c1);
        mesh->c1.addPoint(box.upper.c1);
        mesh->c2.addPoint(box.lower.c2);
        mesh->c2.addPoint(box.upper.c2);
    }

    mesh->setOptimalIterationOrder();
    return mesh;
}


RectilinearMesh1D RectilinearMesh2DDividingGenerator::get1DMesh(const RectilinearMesh1D& initial, const shared_ptr<GeometryElementD<2>>& geometry, size_t dir)
{
    // TODO: Użyj algorytmu Roberta, może będzie lepszy

    RectilinearMesh1D result = initial;

    // First add refinement points
    for (auto ref: refinements[dir]) {
        auto boxes = geometry->getLeafsBoundingBoxes(&ref.first);
        if (warn_multiple && boxes.size() > 1) writelog(LOG_WARNING, "RectilinearMesh2DDividingGenerator: Single refinement defined for more than one object.");
        if (warn_multiple && boxes.size() == 0) writelog(LOG_WARNING, "RectilinearMesh2DDividingGenerator: Refinement defined for object absent from the geometry.");
        for (auto box: boxes) {
            for (auto x: ref.second) {
                if (warn_outside && (x < 0 || x > box.upper[dir]-box.lower[dir]))
                    writelog(LOG_WARNING, "RectilinearMesh2DDividingGenerator: Refinement at %1% outside of the object (0 ... %2%).", x, box.upper[dir]-box.lower[dir]);
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

shared_ptr<RectilinearMesh2D> RectilinearMesh2DDividingGenerator::generate(const shared_ptr<GeometryElementD<2>>& geometry)
{
    RectilinearMesh2D initial;
    std::vector<Box2D> boxes = geometry->getLeafsBoundingBoxes();
    for (auto& box: boxes) {
        initial.c0.addPoint(box.lower.c0);
        initial.c0.addPoint(box.upper.c0);
        initial.c1.addPoint(box.lower.c1);
        initial.c1.addPoint(box.upper.c1);
    }

    auto mesh = make_shared<RectilinearMesh2D>();
    mesh->c0 = get1DMesh(initial.c0, geometry, 0);
    mesh->c1 = get1DMesh(initial.c1, geometry, 1);

    mesh->setOptimalIterationOrder();
    return mesh;
}

} // namespace plask