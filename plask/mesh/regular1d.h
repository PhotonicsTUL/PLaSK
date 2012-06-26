#ifndef PLASK__REGULAR1D_H
#define PLASK__REGULAR1D_H

/** @file
This file defines regular mesh for 1d space.
*/

#include "../utils/iterators.h"
#include "../utils/interpolation.h"


namespace plask {

/**
 * Regular mesh in 1d space.
 */
class RegularMesh1d {

    double lo, step;
    std::size_t points_count;

  public:

    /// Type of points in this mesh.
    typedef double PointType;

    typedef IndexedIterator<RegularMesh1d> iterator;
    typedef IndexedIterator<RegularMesh1d> const_iterator;

    /// @return iterator referring to the first point in this mesh
    const_iterator begin() const { return const_iterator(this, 0); }

    /// @return iterator referring to the past-the-end point in this mesh
    const_iterator end() const { return const_iterator(this, points_count); }

    /// Construct uninitialized mesh.
    RegularMesh1d() {}

    /**
     * Construct mesh with given paramters.
     * @param first coordinate of first point in mesh
     * @param last coordinate of last point in mesh
     * @param points_count number of points in mesh
     */
    RegularMesh1d(double first, double last, std::size_t points_count)
        : lo(first), step((last - first) / (points_count-1)), points_count(points_count) {}

    /**
     * Set new mesh parameters.
     * @param first coordinate of first point in mesh
     * @param last coordinate of last point in mesh
     * @param points_count number of points in mesh
     */
    void reset(double first, double last, std::size_t points_count) {
        lo = first;
        step = (last - first) / (points_count-1);
        points_count = points_count;
    }

    /**
     * @return coordinate of first point in mesh
     */
    double getFirst() const { return lo; }

    /**
     * @return coordinate of last point in mesh
     */
    double getLast() const { return lo + step * (points_count-1); }

    /**
     * @return distanse between two neighbouring points in mesh
     */
    double getStep() const { return step; }

    /// @return number of points in mesh
    std::size_t size() const { return points_count; }

     /**
      * Compares meshes
      * It use algorithm which has contant time complexity.
      * @param to_compare mesh to compare
      * @return @c true only if this mesh and @p to_compare represents the same set of points
      */
     bool operator==(const RegularMesh1d& to_compare) const {
         return this->lo == to_compare.lo && this->step == to_compare.step && this->points_count == to_compare.points_count;
     }

     /**
      * Print mesh to stream
      * @param out stream to print
      * @return out
      */
     friend inline std::ostream& operator<<(std::ostream& out, const RegularMesh1d& self) {
         out << "[";
         for (std::size_t i = 0; i < points_count; ++i) {
             if (i != 0) out << ", ";
             out << this->operator [](i);
         }
         out << "]";
         return out;
     }

     /// @return true only if there are no points in mesh
     bool empty() const { return points_count == 0; }

     /**
      * Get point by index.
      * @param index index of point, from 0 to size()-1
      * @return point with given @p index
      */
     const double& operator[](std::size_t index) const { return lo + index * step; }

     /**
      * Remove all points from mesh.
      */
     void clear() { points_count = 0; }

     /**
      * Find index where @p to_find point could be inserted.
      * @param to_find point to find
      * @return First index where to_find could be inserted.
      *         Refer to value equal to @p to_find only if @p to_find is already in mesh.
      *         Can be equal to size() if to_find is higher than all points in mesh.
      */
     std::size_t findIndex(double to_find) const {
         return clamp(int(cail((to_find - lo) / step)), 0, int(points_count));

     }

     /**
      * Find position where @p to_find point could be inserted.
      * @param to_find point to find
      * @return First position where to_find could be insert.
      *         Refer to value equal to @p to_find only if @p to_find is in mesh.
      *         Can be equal to end() if to_find is higher than all points in mesh
      *         (in such case returned iterator can't be dereferenced).
      */
     const_iterator find(double to_find) const {
         return begin() + findIndex(to_find);
     }

     /**
      * Calculate (using linear interpolation) value of data in point using data in points describe by this mesh.
      * @param data values of data in points describe by this mesh
      * @param point point in which value should be calculate
      * @return interpolated value in point @p point
      */
     template <typename RandomAccessContainer>
     auto interpolateLinear(const RandomAccessContainer& data, double point) -> typename std::remove_reference<decltype(data[0])>::type;

 };

 // RegularMesh1d method templates implementation
 template <typename RandomAccessContainer>
 auto RegularMesh1d::interpolateLinear(const RandomAccessContainer& data, double point) -> typename std::remove_reference<decltype(data[0])>::type {
     std::size_t index = findIndex(point);
     if (index == size()) return data[index - 1];     //TODO what should do if mesh is empty?
     if (index == 0 || points[index] == point) return data[index]; //hit exactly
     // here: points[index-1] < point < points[index]
     return interpolation::linear(points[index-1], data[index-1], points[index], data[index], point);
 }

 }   // namespace plask

#endif // PLASK__REGULAR1D_H
