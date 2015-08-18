#ifndef PLASK__UNORDERED1D_H
#define PLASK__UNORDERED1D_H

/** @file
This file contains rectilinear mesh for 1d space.
*/

#include <vector>
#include <algorithm>
#include <initializer_list>
#include <limits>

#include "mesh.h"
#include "../vec.h"
#include "../utils/iterators.h"
#include "../utils/interpolation.h"
#include "rectangular1d.h"
#include "ordered1d.h"


namespace plask {

/**
 * Rectilinear mesh in 1D space.
 */
class PLASK_API UnorderedAxis: public RectangularAxis {

    /// Points coordinates in ascending order.
    std::vector<double> points;

public:

    /// Type of points in this mesh.
    typedef double PointType;

    /// Random access iterator type which allow iterate over all points in this mesh, in ascending order.
    typedef std::vector<double>::const_iterator native_const_iterator;

    /// @return iterator referring to the first point in this mesh
    native_const_iterator begin() const { return points.begin(); }

    /// @return iterator referring to the past-the-end point in this mesh
    native_const_iterator end() const { return points.end(); }

    /// @return vector of points (reference to internal vector)
    const std::vector<double>& getPointsVector() const { return points; }

    /**
     * Find position nearest to @p to_find.
     * @param to_find
     * @return position pos for which abs(*pos-to_find) is minimal
     */
    native_const_iterator findNearest(double to_find) const;

    /**
     * Find index nearest to @p to_find.
     * @param to_find
     * @return index i for which abs((*this)[i]-to_find) is minimal
     */
    std::size_t findNearestIndex(double to_find) const override { return findNearest(to_find) - begin(); }

    /// Construct an empty mesh.
    UnorderedAxis() {}

    /// Copy constructor. It does not copy the owner.
    UnorderedAxis(const UnorderedAxis& src): points(src.points) {}

    /// Move constructor. It does not move the owner.
    UnorderedAxis(UnorderedAxis&& src): points(std::move(src.points)) {}

    /// Copy constructor from any RectangularAxis
    UnorderedAxis(const RectangularAxis& src): points(src.size()) {
        std::copy(src.begin(), src.end(), points.begin());
    }

    operator OrderedAxis() const {
        return OrderedAxis(points);
    }
    
    /**
     * Construct mesh with given points.
     * It use algorithm which has logarithmic time complexity.
     * @param points points, in any order
     */
    UnorderedAxis(std::initializer_list<PointType> points);

    /**
     * Construct mesh with points given in a vector.
     * It use algorithm which has logarithmic time complexity pew point in @p points.
     * @param points points, in any order
     */
    UnorderedAxis(const std::vector<PointType>& points);

    /**
     * Construct mesh with points given in a vector.
     * It use algorithm which has logarithmic time complexity pew point in @p points.
     * @param points points, in any order
     */
    UnorderedAxis(std::vector<PointType>&& points);

    /// Assign a new mesh. This operation preserves the \a owner.
    UnorderedAxis& operator=(const UnorderedAxis& src);

    /// Assign a new mesh. This operation preserves the \a owner.
    UnorderedAxis& operator=(UnorderedAxis&& src);

    /// Assign a new mesh. This operation preserves the \a owner.
    UnorderedAxis& operator=(const RectangularAxis& src);

    /**
     * Compares meshes
     * It use algorithm which has linear time complexity.
     * @param to_compare mesh to compare
     * @return @c true only if this mesh and @p to_compare represents the same set of points
     */
    bool operator==(const UnorderedAxis& to_compare) const;

    void writeXML(XMLElement& object) const override;

    /// @return number of points in mesh
    virtual std::size_t size() const override { return points.size(); }

    virtual double at(std::size_t index) const override { return points[index]; }

    // @return true only if there are no points in mesh
    //bool empty() const { return points.empty(); }

    /**
     * Reserve space for new points
     * \param size total size to reserve
     */
    void reserve(size_t size) { points.reserve(size); }
    
    /**
     * Append (1d) point to this mesh.
     * Point is added to mesh only if it is not closer than the specified minimum distance to the existing one.
     * It use algorithm which has O(size()) time complexity.
     * \param new_node_cord coordinate of point to add
     */
    void appendPoint(double new_node_cord);

    /**
     * Insert (1d) point to this mesh.
     * Point is added to mesh only if it is not closer than the specified minimum distance to the existing one.
     * It use algorithm which has O(size()) time complexity.
     * \param index index of the new inserted point
     * \param new_node_cord coordinate of point to add
     */
    void insertPoint(size_t index, double new_node_cord);

    /**
     * Remove point at specified index
     * \param index intex of the point to remove
     */
    void removePoint(std::size_t index);

    /**
     * Append points from ordered range.
     * It uses algorithm which has linear time complexity.
     * @param begin, end ordered range of points in ascending order
     * @tparam IteratorT input iterator
     */
    template <typename IteratorT>
    void appendPoints(IteratorT begin, IteratorT end);

    /**
     * Remove all points from mesh.
     */
    void clear();

    shared_ptr<RectangularAxis> clone() const override;

    bool isIncreasing() const override { return false; }

    /**
     * Calculate (using linear interpolation) value of data in point using data in points described by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, double point) const -> typename std::remove_reference<decltype(data[0])>::type;

};

// OrderedAxis method templates implementation
template <typename RandomAccessContainer>
inline auto UnorderedAxis::interpolateLinear(const RandomAccessContainer& data, double point) const -> typename std::remove_reference<decltype(data[0])>::type {
    double left = std::numeric_limits<double>::min(), right = std::numeric_limits<double>::max();
    double minv = std::numeric_limits<double>::max(), maxv = std::numeric_limits<double>::min();
    size_t ileft = points.size(), iright = points.size(), imin, imax;
    for (size_t i = 0; i != points.size(); ++i) {
        auto p = points[i];
        if (maxv < p) { maxv = p; imax = i; }
        if (p < minv) { minv = p; imin = i; }
        if (left < p && p <= point) { left = p; ileft = i; }
        if (point <= p && p < right) { right = p; iright = i; }
    }
    if (left == right) return data[ileft];
    if (ileft == points.size()) return data[imin];
    if (iright == points.size()) return data[imax];
    return interpolation::linear(left, data[ileft], right, data[iright], point);
}

}   // namespace plask

#endif // PLASK__UNORDERED1D_H
