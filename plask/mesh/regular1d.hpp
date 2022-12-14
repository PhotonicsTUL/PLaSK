/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#ifndef PLASK__REGULAR1D_H
#define PLASK__REGULAR1D_H

/** @file
This file defines regular mesh for 1d space.
*/

#include "mesh.hpp"
#include "../math.hpp"
#include "../utils/iterators.hpp"
#include "../utils/interpolation.hpp"
#include "../utils/stl.hpp"

#include "axis1d.hpp"

namespace plask {

/**
 * Regular mesh in 1d space.
 */
class PLASK_API RegularAxis: public MeshAxis {

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

    /// Construct uninitialized mesh.
    RegularAxis():
        lo(0.), _step(0.), points_count(0) {}

    /// Copy constructor. It does not copy owner
    RegularAxis(const RegularAxis& src):
        lo(src.lo), _step(src._step), points_count(src.points_count) {}

    /**
     * Construct mesh with given paramters.
     * @param first coordinate of first point in mesh
     * @param last coordinate of last point in mesh
     * @param points_count number of points in mesh
     */
    RegularAxis(double first, double last, std::size_t points_count):
        lo(first), _step( (last - first) / ((points_count>1)?double(points_count-1):1.) ),
        points_count(points_count) {}

    /// Assign a new mesh. This operation preserves the \a owner.
    RegularAxis& operator=(const RegularAxis& src);

    /**
     * Set new mesh parameters.
     * @param first coordinate of first point in mesh
     * @param last coordinate of last point in mesh
     * @param points_count number of points in mesh
     */
    void reset(double first, double last, std::size_t points_count);

    /**
     * @return coordinate of the first point in the mesh
     */
    double first() const { return lo; }

    /**
     * @return coordinate of the last point in the mesh
     */
    double last() const { return lo + _step * double(points_count-1); }

    /**
     * @return distance between two neighboring points in the mesh
     */
    double step() const { return _step; }

    /// @return number of points in the mesh
    std::size_t size() const override { return points_count; }

    /**
     * Compare meshes.
     * It uses algorithm which has constant time complexity.
     * @param to_compare mesh to compare
     * @return @c true only if this mesh and @p to_compare represents the same set of points
     */
    bool operator==(const RegularAxis& to_compare) const {
        return this->lo == to_compare.lo && this->_step == to_compare._step && this->points_count == to_compare.points_count;
    }

    /**
     * Compare meshes.
     * It uses algorithm which has constant time complexity.
     * @param to_compare mesh to compare
     * @return @c true only if this mesh and @p to_compare represents different sets of points
     */
    bool operator!=(const RegularAxis& to_compare) const {
        return !(*this == to_compare);
    }

    /// @return true only if there are no points in mesh
    bool empty() const override { return points_count == 0; }

    /**
     * Get point by index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    double operator[](std::size_t index) const { return lo + double(index) * _step; }

    double at(std::size_t index) const override { return lo + double(index) * _step; }

    /**
     * Remove all points from mesh.
     */
    void clear() {
        if (empty()) return;
        points_count = 0;
        fireResized();
    }

    std::size_t findIndex(double to_find) const override {
        return clamp(std::ptrdiff_t(std::ceil((to_find - lo) / _step)), std::ptrdiff_t(0), std::ptrdiff_t(points_count));
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

    std::size_t findUpIndex(double to_find) const override;

    /**
     * Find the lowest position for with a coordinate larger than @p to_find.
     * @param to_find point to find
     * @return The first position with coordinate larger than @p to_find.
     *         It equals to end() if @p to_find is larger than all points in mesh or equals to the last point.
     */
    native_const_iterator findUp(double to_find) const {
        return begin() + findUpIndex(to_find);
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
    std::size_t findNearestIndex(double to_find) const override { return findNearest(to_find) - begin(); }

    shared_ptr<MeshAxis> clone() const override { return plask::make_shared<RegularAxis>(*this); }

    void writeXML(XMLElement& object) const override;

    bool isIncreasing() const override;

    shared_ptr<MeshAxis> getMidpointAxis() const override;

    /**
     * Calculate (using linear interpolation) value of data in point using data in points describe by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, double point) -> typename std::remove_reference<decltype(data[0])>::type;

protected:
    bool hasSameNodes(const MeshD<1>& to_compare) const override;

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

typedef RegularAxis RegularMesh1D;

PLASK_API shared_ptr<RegularMesh1D> readRegularMeshAxis(XMLReader& reader);

}   // namespace plask

#endif // PLASK__REGULAR1D_H
