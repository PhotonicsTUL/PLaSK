#ifndef PLASK__RECTILINEAR1D_H
#define PLASK__RECTILINEAR1D_H

/** @file
This file includes rectilinear mesh for 1d space.
*/

#include <vector>
#include <algorithm>
#include <initializer_list>

#include "mesh.h"
#include "../vec.h"
#include "../utils/iterators.h"
#include "../utils/interpolation.h"

namespace plask {

/**
 * Rectilinear mesh in 1D space.
 */
class RectilinearMesh1D {

    /// Points coordinates in ascending order.
    std::vector<double> points;

    void sortPointsAndRemoveNonUnique();

public:

    /// Type of points in this mesh.
    typedef double PointType;

    /// Random access iterator type which allow iterate over all points in this mesh, in ascending order.
    typedef std::vector<double>::const_iterator const_iterator;

    /// Pointer to mesh holding this axis
    Mesh* owner;

    /// @return iterator referring to the first point in this mesh
    const_iterator begin() const { return points.begin(); }

    /// @return iterator referring to the past-the-end point in this mesh
    const_iterator end() const { return points.end(); }

    /// @return vector of points (reference to internal vector)
    const std::vector<double>& getPointsVector() const { return points; }

    /**
     * Find position where @p to_find point could be inserted.
     * @param to_find point to find
     * @return First position where to_find could be insert.
     *         Refer to value equal to @p to_find only if @p to_find is already in mesh.
     *         Can be equal to end() if to_find is higher than all points in mesh
     *         (in such case returned iterator can't be dereferenced).
     */
    const_iterator find(double to_find) const;

    /**
     * Find index where @p to_find point could be inserted.
     * @param to_find point to find
     * @return First index where to_find could be inserted.
     *         Refer to value equal to @p to_find only if @p to_find is already in mesh.
     *         Can be equal to size() if to_find is higher than all points in mesh.
     */
    std::size_t findIndex(double to_find) const { return find(to_find) - begin(); }

    /**
     * Find position nearest to @p to_find.
     * @param to_find
     * @return position pos for which abs(*pos-to_find) is minimal
     */
    const_iterator findNearest(double to_find) const;

    /**
     * Find index nearest to @p to_find.
     * @param to_find
     * @return index i for which abs((*this)[i]-to_find) is minimal
     */
    std::size_t findNearestIndex(double to_find) const { return findNearest(to_find) - begin(); }

    //should we allow for non-const iterators?
    /*typedef std::vector<double>::iterator iterator;
    iterator begin() { return points.begin(); }
    iterator end() { return points.end(); }*/

    /// Construct an empty mesh.
    RectilinearMesh1D(): owner(nullptr) {}

    /// Copy constructor. It does not copy owner.
    RectilinearMesh1D(const RectilinearMesh1D& src): points(src.points), owner(nullptr) {}

    /// Move constructor. It does not move owner.
    RectilinearMesh1D(RectilinearMesh1D&& src): points(std::move(src.points)), owner(nullptr) {}

    /**
     * Construct mesh with given points.
     * It use algorithm which has logarithmic time complexity.
     * @param points points, in any order
     */
    RectilinearMesh1D(std::initializer_list<PointType> points);

    /**
     * Construct mesh with points given in a vector.
     * It use algorithm which has logarithmic time complexity pew point in @p points.
     * @param points points, in any order
     */
    RectilinearMesh1D(const std::vector<PointType>& points);

    /**
     * Construct mesh with points given in a vector.
     * It use algorithm which has logarithmic time complexity pew point in @p points.
     * @param points points, in any order
     */
    RectilinearMesh1D(std::vector<PointType>&& points);

    /// Assign a new mesh. This operation preserves the \a owner.
    RectilinearMesh1D& operator=(const RectilinearMesh1D& src) {
        bool resized = size() != src.size();
        points = src.points;
        if (owner) {
            if (resized) owner->fireResized();
            else owner->fireChanged();
        }
        return *this;
    }

    /// Assign a new mesh. This operation preserves the \a owner.
    RectilinearMesh1D& operator=(RectilinearMesh1D&& src) {
        bool resized = size() != src.size();
        std::swap(points, src.points);
        if (owner) {
            if (resized) owner->fireResized();
            else owner->fireChanged();
        }
        return *this;
    }

    /**
     * Compares meshes
     * It use algorithm which has linear time complexity.
     * @param to_compare mesh to compare
     * @return @c true only if this mesh and @p to_compare represents the same set of points
     */
    bool operator==(const RectilinearMesh1D& to_compare) const;

    /**
     * Print mesh to stream
     * @param out stream to print
     * @param mesh mesh to print
     * @return out
     */
    friend inline std::ostream& operator<<(std::ostream& out, const RectilinearMesh1D& mesh) {
        out << "[";
        for (auto p: mesh.points) {
            out << p << ((p != mesh.points.back())? ", " : "");
        }
        out << "]";
        return out;
    }

    /// @return number of points in mesh
    std::size_t size() const { return points.size(); }

    /// @return true only if there are no points in mesh
    bool empty() const { return points.empty(); }

    /**
     * Add (1d) point to this mesh.
     * Point is added to mesh only if it is not already included in it.
     * It use algorithm which has O(size()) time complexity.
     * @param new_node_cord coordinate of point to add
     */
    void addPoint(double new_node_cord);

    /**
     * Remove point at specified index
     * \param index intex of the point to remove
     */
    void removePoint(std::size_t index);

    /**
     * Add points from ordered range.
     * It uses algorithm which has linear time complexity.
     * @param begin, end ordered range of points in ascending order
     * @param points_count_hint number of points in range (can be approximate, or 0)
     * @tparam IteratorT input iterator
     */
    template <typename IteratorT>
    void addOrderedPoints(IteratorT begin, IteratorT end, std::size_t points_count_hint);

    /**
     * Add points from ordered range.
     * It uses algorithm which has linear time complexity.
     * @param begin, end ordered range of points in ascending order
     * @tparam RandomAccessIteratorT random access iterator
     */
    //TODO use iterator traits and write version for input iterator
    template <typename RandomAccessIteratorT>
    void addOrderedPoints(RandomAccessIteratorT begin, RandomAccessIteratorT end) { addOrderedPoints(begin, end, end - begin); }

    /**
     * Add to mesh points: first + i * (last-first) / points_count, where i is in range [0, points_count].
     * It uses algorithm which has linear time complexity.
     * @param first coordinate of the first point
     * @param last coordinate of the last point
     * @param points_count number of points to add
     */
    void addPointsLinear(double first, double last, std::size_t points_count);

    /**
     * Get point by index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    const double& operator[](std::size_t index) const { return points[index]; }

    /**
     * Remove all points from mesh.
     */
    void clear();

    /**
     * Calculate (using linear interpolation) value of data in point using data in points described by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, double point) const -> typename std::remove_reference<decltype(data[0])>::type;

};

// RectilinearMesh1D method templates implementation
template <typename RandomAccessContainer>
auto RectilinearMesh1D::interpolateLinear(const RandomAccessContainer& data, double point) const -> typename std::remove_reference<decltype(data[0])>::type {
    std::size_t index = findIndex(point);
    if (index == size()) return data[index - 1];     //TODO what should it do if mesh is empty?
    if (index == 0 || points[index] == point) return data[index]; // hit exactly
    // here: points[index-1] < point < points[index]
    return interpolation::linear(points[index-1], data[index-1], points[index], data[index], point);
}

template <typename IteratorT>
inline void RectilinearMesh1D::addOrderedPoints(IteratorT begin, IteratorT end, std::size_t points_count_hint) {
    std::vector<double> result;
    result.reserve(this->size() + points_count_hint);
    std::set_union(this->points.begin(), this->points.end(), begin, end, std::back_inserter(result));
    this->points = std::move(result);
    if (owner) owner->fireResized();
};

}   // namespace plask

#endif // PLASK__RECTILINEAR1D_H
