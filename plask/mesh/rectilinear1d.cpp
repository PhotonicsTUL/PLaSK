#include "rectilinear1d.h"

namespace plask {

RectilinearMesh1d::RectilinearMesh1d(std::initializer_list< PointType > points) {
    for (PointType p: points) addPoint(p);
}

bool RectilinearMesh1d::operator==(const plask::RectilinearMesh1d& to_compare) const {
    return points == to_compare.points;
}

RectilinearMesh1d::const_iterator RectilinearMesh1d::find(double to_find) const {
    return std::lower_bound(points.begin(), points.end(), to_find);
}

void RectilinearMesh1d::addPoint(double new_node_cord) {
    auto where = std::lower_bound(points.begin(), points.end(), new_node_cord);
    if (where == points.end())
        points.push_back(new_node_cord);
    else
        if (*where != new_node_cord)    // if node not already included
            points.insert(where, new_node_cord);
}

void RectilinearMesh1d::addPointsLinear(double first, double last, std::size_t points_count) {
    if (points_count == 0) return;
    --points_count;
    double len = last - first;
    auto get_el = [&](std::size_t i) { return first + i * len / points_count; };
    addOrderedPoints(makeFunctorIndexedIterator(get_el, 0), makeFunctorIndexedIterator(get_el, points_count+1), points_count+1);
}

void RectilinearMesh1d::clear() {
    points.clear();
}

}   // namespace plask
