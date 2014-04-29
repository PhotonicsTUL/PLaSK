#ifndef PLASK__REGULAR1D_H
#define PLASK__REGULAR1D_H

/** @file
This file defines regular mesh for 1d space.
*/

#include "mesh.h"
#include "../math.h"
#include "../utils/iterators.h"
#include "../utils/interpolation.h"
#include "../utils/stl.h"

#include "rectangular1d.h"

namespace plask {

/**
 * Regular mesh in 1d space.
 */
class RegularAxis: public RectangularAxis {

    double lo, _step;
    std::size_t points_count;

  public:

    /// Type of points in this mesh.
    typedef double PointType;

    typedef IndexedIterator<const RegularAxis, PointType> native_const_iterator;

    /// @return iterator referring to the first point in this mesh
    native_const_iterator begin() const { return native_const_iterator(this, 0); }

    /// @return iterator referring to the past-the-end point in this mesh
    native_const_iterator end() const { return native_const_iterator(this, points_count); }

    /// Pointer to mesh holding this axis
    Mesh* owner;

    /// Construct uninitialized mesh.
    RegularAxis():
        lo(0.), _step(0.), points_count(0), owner(nullptr) {}

    /// Copy constructor. It does not copy owner
    RegularAxis(const RegularAxis& src):
        lo(src.lo), _step(src._step), points_count(src.points_count), owner(nullptr) {}

    /**
     * Construct mesh with given paramters.
     * @param first coordinate of first point in mesh
     * @param last coordinate of last point in mesh
     * @param points_count number of points in mesh
     */
    RegularAxis(double first, double last, std::size_t points_count):
        lo(first), _step( (last - first) / ((points_count>1)?(points_count-1):1.) ),
        points_count(points_count), owner(nullptr) {}

    /// Assign a new mesh. This operation preserves the \a owner.
    RegularAxis& operator=(const RegularAxis& src) {
        bool resized = points_count != src.points_count;
        lo = src.lo; _step = src._step; points_count = src.points_count;
        if (owner) {
            if (resized) owner->fireResized();
            else owner->fireChanged();
        }
        return *this;
    }

    /**
     * Set new mesh parameters.
     * @param first coordinate of first point in mesh
     * @param last coordinate of last point in mesh
     * @param points_count number of points in mesh
     */
    void reset(double first, double last, std::size_t points_count) {
        lo = first;
        _step = (last - first) / ((points_count>1)?(points_count-1):1.);
        bool resized = this->points_count != points_count;
        this->points_count = points_count;
        if (owner) {
            if (resized) owner->fireResized();
            else owner->fireChanged();
        }
    }

    /**
     * @return coordinate of the first point in the mesh
     */
    double first() const { return lo; }

    /**
     * @return coordinate of the last point in the mesh
     */
    double last() const { return lo + _step * (points_count-1); }

    /**
     * @return distance between two neighboring points in the mesh
     */
    double step() const { return _step; }

    /// @return number of points in the mesh
    virtual std::size_t size() const override { return points_count; }

    /**
     * Compare meshes
     * It uses algorithm which has constant time complexity.
     * @param to_compare mesh to compare
     * @return @c true only if this mesh and @p to_compare represents the same set of points
     */
    bool operator==(const RegularAxis& to_compare) const {
        return this->lo == to_compare.lo && this->_step == to_compare._step && this->points_count == to_compare.points_count;
    }

    /// @return true only if there are no points in mesh
    bool empty() const { return points_count == 0; }

    /**
     * Get point by index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    const double operator[](std::size_t index) const { return lo + index * _step; }

    virtual double at(std::size_t index) const override { return lo + index * _step; }

    /**
     * Remove all points from mesh.
     */
    void clear() {
        points_count = 0;
        if (owner) owner->fireResized();
    }

    /**
     * Find index where @p to_find point could be inserted.
     * @param to_find point to find
     * @return First index where to_find could be inserted.
     *         Refer to value equal to @p to_find only if @p to_find is already in mesh, in other case it refer to value bigger than to_find.
     *         Can be equal to size() if to_find is higher than all points in mesh.
     */
    std::size_t findIndex(double to_find) const {
        return clamp(int(std::ceil((to_find - lo) / _step)), 0, int(points_count));
    }

    /**
     * Find position where @p to_find point could be inserted.
     * @param to_find point to find
     * @return First position where to_find could be insert.
     *         Refer to value equal to @p to_find only if @p to_find is already in mesh, in other case it refer to value bigger than to_find.
     *         Can be equal to end() if to_find is higher than all points in mesh
     *         (in such case returned iterator can't be dereferenced).
     */
    native_const_iterator find(double to_find) const {
        return begin() + findIndex(to_find);
    }

    /**
     * Find position nearest to @p to_find.
     * @param to_find
     * @return position pos for which abs(*pos-to_find) is minimal
     */
    native_const_iterator findNearest(double to_find) const {
        return find_nearest_using_lower_bound(begin(), end(), to_find, find(to_find));
    }

    /**
     * Find index nearest to @p to_find.
     * @param to_find
     * @return index i for which abs((*this)[i]-to_find) is minimal
     */
    std::size_t findNearestIndex(double to_find) const { return findNearest(to_find) - begin(); }

    virtual shared_ptr<RectangularMesh<1>> clone() const override { return make_shared<RegularAxis>(*this); }

    void writeXML(XMLElement& object) const override;

    bool isIncreasing() const override;

    /**
     * Calculate (using linear interpolation) value of data in point using data in points describe by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, double point) -> typename std::remove_reference<decltype(data[0])>::type;

};

// RegularAxis method templates implementation
template <typename RandomAccessContainer>
auto RegularAxis::interpolateLinear(const RandomAccessContainer& data, double point) -> typename std::remove_reference<decltype(data[0])>::type {
    std::size_t index = findIndex(point);
    if (index == size()) return data[index - 1];     //TODO what should do if mesh is empty?
    if (index == 0 || this->operator[](index) == point) return data[index]; //hit exactly
    // here: points[index-1] < point < points[index]
    return interpolation::linear(this->operator[](index-1), data[index-1], this->operator[](index), data[index], point);
}

}   // namespace plask

#endif // PLASK__REGULAR1D_H
