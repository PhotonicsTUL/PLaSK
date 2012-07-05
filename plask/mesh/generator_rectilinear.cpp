#include <deque>

#include "generator_rectilinear.h"

namespace plask {

RectilinearMesh1D RectilinearMesh2DfromSimpleDivision::get1DMesh(const RectilinearMesh1D& initial)
{
    // TODO: Użyj algorytmu Roberta, może będzie lepszy

    RectilinearMesh1D result = initial;

    // First divide each element
    double x = *initial.begin();
    for (auto i = initial.begin()+1; i!= initial.end(); ++i) {
        double w = *i - x;
        std::vector<double> points; points.reserve(division-1);
        for (size_t j = 1; j != division; ++j) points.push_back(x + w*j/division);
        result.addOrderedPoints(points.begin(), points.end());
        x = *i;
    }

    if (result.size() < 3) return result;

    bool repeat;
    do {
        double w_prev = INFINITY, w = result[1]-result[0], w_next = result[2]-result[1];
        repeat = false;
        for (size_t i = 1; i != result.size(); ++i) {
            if (w > 2.*w_prev || w > 2.*w_next) {
                result.addPoint(0.5 * (result[i-1]+result[i]));
                repeat = true;
                break;
            }
            w_prev = w;
            w = w_next;
            w_next = (i >= result.size()-2)? INFINITY : result[i+2] - result[i+1];
        }
    } while (repeat);
    return result;
}

shared_ptr<RectilinearMesh2D> RectilinearMesh2DfromSimpleDivision::generate(const shared_ptr<GeometryElementD<2>>& geometry)
{
    RectilinearMesh2D initial = RectilinearMeshFromGeometry(geometry);
    auto mesh = make_shared<RectilinearMesh2D>();
    mesh->c0 = get1DMesh(initial.c0);
    mesh->c1 = get1DMesh(initial.c1);
    return mesh;
}

} // namespace plask