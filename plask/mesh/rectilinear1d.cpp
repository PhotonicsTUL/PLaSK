#include "rectilinear1d.h"

#include "../utils/stl.h"

namespace plask {

void RectilinearMesh1D::sortPointsAndRemoveNonUnique()
{
    std::sort(this->points.begin(), this->points.end());
    this->points.erase(std::unique(this->points.begin(), this->points.end()), this->points.end());
}

RectilinearMesh1D::RectilinearMesh1D(std::initializer_list<PointType> points): points(points), owner(nullptr) {
    sortPointsAndRemoveNonUnique();
}

RectilinearMesh1D::RectilinearMesh1D(std::vector<PointType> &&points): points(std::move(points)), owner(nullptr) {
    sortPointsAndRemoveNonUnique();
}

RectilinearMesh1D::RectilinearMesh1D(const std::vector<PointType> &points) : points(points), owner(nullptr) {
    sortPointsAndRemoveNonUnique();
}

bool RectilinearMesh1D::operator==(const plask::RectilinearMesh1D& to_compare) const {
    return points == to_compare.points;
}



RectilinearMesh1D::const_iterator RectilinearMesh1D::find(double to_find) const {
    return std::lower_bound(points.begin(), points.end(), to_find);
}

RectilinearMesh1D::const_iterator RectilinearMesh1D::findNearest(double to_find) const {
    return find_nearest_binary(points.begin(), points.end(), to_find);
}

void RectilinearMesh1D::addPoint(double new_node_cord) {
    auto where = std::lower_bound(points.begin(), points.end(), new_node_cord);
    if (where == points.end())
        points.push_back(new_node_cord);
    else
        if (*where != new_node_cord)    // if node not already included
            points.insert(where, new_node_cord);
    if (owner) owner->fireResized();
}

void RectilinearMesh1D::addPointsLinear(double first, double last, std::size_t points_count) {
    if (points_count == 0) return;
    --points_count;
    double len = last - first;
    auto get_el = [&](std::size_t i) { return first + i * len / points_count; };
    addOrderedPoints(makeFunctorIndexedIterator(get_el, 0), makeFunctorIndexedIterator(get_el, points_count+1), points_count+1);
    if (owner) owner->fireResized();
}

void RectilinearMesh1D::removePoint(std::size_t index) {
    points.erase(points.begin() + index);
    if (owner) owner->fireResized();
}


void RectilinearMesh1D::clear() {
    points.clear();
    if (owner) owner->fireResized();
}

}   // namespace plask
