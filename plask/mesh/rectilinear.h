#ifndef PLASK__RECTILINEAR_H
#define PLASK__RECTILINEAR_H

/** @file
This file includes rectilinear meshes for 1d, 2d, and 3d spaces.
*/

#include <vector>
#include <algorithm>
#include <initializer_list>

#include "../vec.h"
#include "../utils/iterators.h"
#include "../utils/interpolation.h"

namespace plask {

/**
 * Rectilinear mesh in 1d space.
 */
class RectilinearMesh1d {

    ///Points coordinates in ascending order.
    std::vector<double> points;

public:

    ///Type of points in this mesh.
    typedef double PointType;

    ///Random access iterator type which allow iterate over all points in this mesh, in ascending order.
    typedef std::vector<double>::const_iterator const_iterator;

    ///@return iterator referring to the first point in this mesh
    const_iterator begin() const { return points.begin(); }

    ///@return iterator referring to the past-the-end point in this mesh
    const_iterator end() const { return points.end(); }

    /**
     * Find position where @a to_find point could be insert.
     * @param to_find point to find
     * @return First position where to_find could be insert.
     *         Refer to value equal to @a to_find only if @a to_find is already in mesh.
     *         Can be equal to end() if to_find is higher than all points in mesh
     *         (in such case returned iterator can't be dereferenced).
     */
    const_iterator find(double to_find) const;

    /**
     * Find index where @a to_find point could be insert.
     * @param to_find point to find
     * @return First index where to_find could be insert.
     *         Refer to value equal to @a to_find only if @a to_find is already in mesh.
     *         Can be equal to size() if to_find is higher than all points in mesh.
     */
    std::size_t findIndex(double to_find) const { return find(to_find) - begin(); }

    //should we allow for non-const iterators?
    /*typedef std::vector<double>::iterator iterator;
    iterator begin() { return points.begin(); }
    iterator end() { return points.end(); }*/

    ///Construct empty mesh.
    RectilinearMesh1d() {}

    /**
     * Construct mesh with given points.
     * It use algorithm which has quadric time complexity.
     * @param points points, in any order
     */
    RectilinearMesh1d(std::initializer_list<PointType> points);

    /**
     * Compares meshes.
     * It use algorithm which has linear time complexity.
     * @param to_compare mesh to compare
     * @return @c true only if this mesh and @a to_compare represents the same set of points
     */
    bool operator==(const RectilinearMesh1d& to_compare) const;

    ///@return number of points in mesh
    std::size_t size() const { return points.size(); }

    ///@return true only if there are no points in mesh
    bool empty() const { return points.empty(); }

    /**
     * Add (1d) point to this mesh.
     * Point is add to mesh only if it is not already included in mesh.
     * It use algorithm which has O(size()) time complexity.
     * @param new_node_cord coordinate of point to add
     */
    void addPoint(double new_node_cord);

    /**
     * Add points from ordered range.
     * It use algorithm which has linear time complexity.
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
     * @return point with given @a index
     */
    const double& operator[](std::size_t index) const { return points[index]; }

    /**
     * Remove all points from mesh.
     */
    void clear();

    /**
     * Calculate (using linear interpolation) value of data in point using data in points describe by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @a point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, double point) -> typename std::remove_reference<decltype(data[0])>::type;

};

//RectilinearMesh1d method templates implementation
template <typename RandomAccessContainer>
auto RectilinearMesh1d::interpolateLinear(const RandomAccessContainer& data, double point) -> typename std::remove_reference<decltype(data[0])>::type {
    std::size_t index = findIndex(point);
    if (index == size()) return data[index - 1];     //TODO what should do if mesh is empty?
    if (index == 0 || points[index] == point) return data[index]; //hit exactly
    //here: points[index-1] < point < points[index]
    return interpolation::linear(points[index-1], data[index-1], points[index], data[index], point);
}

template <typename IteratorT>
inline void RectilinearMesh1d::addOrderedPoints(IteratorT begin, IteratorT end, std::size_t points_count_hint) {
    std::vector<double> result;
    result.reserve(this->size() + points_count_hint);
    std::set_union(this->points.begin(), this->points.end(), begin, end, std::back_inserter(result));
    this->points = std::move(result);
};

/**
 * Rectilinear mesh in 2d space.
 *
 * Includes two 1d rectilinear meshes:
 * - c0 (alternative names: tran(), ee_x(), r())
 * - c1 (alternative names: up(), ee_y(), z())
 * Represent all points (x, y) such that x is in c0 and y is in c1.
 */
struct RectilinearMesh2d {

    /**
     * Class which allow to access and iterate over RectilinearMesh2d points in choosen order:
     * - natural one: (c0[0], c1[0]), (c0[1], c1[0]), ..., (c0[c0.size-1], c1[0]), (c0[0], c1[1]), ..., (c0[c0.size()-1], c1[c1.size()-1])
     * - or changed: (c0[0], c1[0]), (c0[0], c1[1]), ..., (c0[0], y[c1.size-1]), (c0[1], c1[0]), ..., (c0[c0.size()-1], c1[c1.size()-1])
     */
    struct Accessor {

        ///Mesh for which we want to have access.
        const RectilinearMesh2d& mesh;

        ///Is order changed?
        bool changed;

        typedef Vec<2,double> PointType;

        typedef IndexedIterator< const Accessor, PointType > const_iterator;

        Accessor(const RectilinearMesh2d& mesh, bool changed): mesh(mesh), changed(changed) {}

        ///@return iterator referring to the first point in this mesh
        const_iterator begin() const { return const_iterator(this, 0); }

        ///@return iterator referring to the past-the-end point in this mesh
        const_iterator end() const { return const_iterator(this, size()); }

        /**
        * Get point with given mesh index.
        * Points are in order depends from changed flag:
        * - if its set: (c0[0], c1[0]), (c0[0], c1[1]), ..., (c0[0], y[c1.size-1]), (c0[1], c1[0]), ..., (c0[c0.size()-1], c1[c1.size()-1])
        * - if its not set: (c0[0], c1[0]), (c0[1], c1[0]), ..., (c0[c0.size-1], c1[0]), (c0[0], c1[1]), ..., (c0[c0.size()-1], c1[c1.size()-1])
        * @param index index of point, from 0 to size()-1
        * @return point with given @a index
        */
        Vec<2,double> operator[](std::size_t index) const {
            if (changed) {
                const std::size_t y_size = mesh.c1.size();
                return Vec<2,double>(mesh.c0[index / y_size], mesh.c1[index % y_size]);
            } else {
                return mesh[index];
            }
        }

        /**
         * Get number of points in mesh.
         * @return number of points in mesh
         */
        std::size_t size() const { return mesh.size(); }

        /**
         * Calculate mesh index using indexes of c0 and c1.
         * @param c0_index index of c0, from 0 to c0.size()-1
         * @param c1_index index of c1, from 0 to c1.size()-1
         * @return this mesh index, from 0 to size()-1
         */
        std::size_t index(std::size_t c0_index, std::size_t c1_index) const {
            return changed ? mesh.c1.size() * c0_index + c1_index : mesh.index(c0_index, c1_index);
        }

        /**
        * Calculate index of c0 using mesh index.
        * @param mesh_index this mesh index, from 0 to size()-1
        * @return index of c0, from 0 to c0.size()-1
        */
        std::size_t index0(std::size_t mesh_index) const {
            return changed ? mesh_index / mesh.c1.size() : index0(mesh_index);
        }

        /**
        * Calculate index of c1 using mesh index.
        * @param mesh_index this mesh index, from 0 to size()-1
        * @return index of c1, from 0 to c1.size()-1
        */
        std::size_t index1(std::size_t mesh_index) const {
            return changed ? mesh_index % mesh.c1.size() : index1(mesh_index);
        }

    };

    ///First coordinate of points in this mesh.
    RectilinearMesh1d c0;

    ///Second coordinate of points in this mesh.
    RectilinearMesh1d c1;

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    RectilinearMesh1d& tran() { return c0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const RectilinearMesh1d& tran() const { return c0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    RectilinearMesh1d& up() { return c1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const RectilinearMesh1d& up() const { return c1; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    RectilinearMesh1d& ee_x() { return c0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const RectilinearMesh1d& ee_x() const { return c0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    RectilinearMesh1d& ee_y() { return c1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const RectilinearMesh1d& ee_y() const { return c1; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    RectilinearMesh1d& rad_r() { return c0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const RectilinearMesh1d& rad_r() const { return c0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    RectilinearMesh1d& rad_z() { return c1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const RectilinearMesh1d& rad_z() const { return c1; }

    ///Type of points in this mesh.
    typedef Vec<2,double> PointType;

    /**
     * Random access iterator type which allow iterate over all points in this mesh, in order appointed by operator[].
     * This iterator type is indexed, which mean that it have (read-write) index field equal to 0 for begin() and growing up to size() for end().
     */
    typedef IndexedIterator< const RectilinearMesh2d, PointType > const_iterator;

    ///@return iterator referring to the first point in this mesh
    const_iterator begin() const { return const_iterator(this, 0); }

    ///@return iterator referring to the past-the-end point in this mesh
    const_iterator end() const { return const_iterator(this, size()); }

    /**
     * Get number of points in mesh.
     * @return number of points in mesh
     */
    std::size_t size() const { return c0.size() * c1.size(); }

    ///@return true only if there are no points in mesh
    bool empty() const { return c0.empty() || c1.empty(); }

    /**
     * Calculate this mesh index using indexes of c0 and c1.
     * @param c0_index index of c0, from 0 to c0.size()-1
     * @param c1_index index of c1, from 0 to c1.size()-1
     * @return this mesh index, from 0 to size()-1
     */
    std::size_t index(std::size_t c0_index, std::size_t c1_index) const {
        return c0_index + c0.size() * c1_index;
    }

    /**
     * Calculate index of c0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of c0, from 0 to c0.size()-1
     */
    std::size_t index0(std::size_t mesh_index) const {
        return mesh_index % c0.size();
    }

    /**
     * Calculate index of y using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of c1, from 0 to c1.size()-1
     */
    std::size_t index1(std::size_t mesh_index) const {
        return mesh_index / c0.size();
    }

    /**
     * Get point with given mesh index.
     * Points are in order: (c0[0], c1[0]), (c0[1], c1[0]), ..., (c0[c0.size-1], c1[0]), (c0[0], c1[1]), ..., (c0[c0.size()-1], c1[c1.size()-1])
     * @param index index of point, from 0 to size()-1
     * @return point with given @a index
     */
    Vec<2,double> operator[](std::size_t index) const {
        const std::size_t c0_size = c0.size();
        return Vec<2, double>(c0[index % c0_size], c1[index / c0_size]);
    }

    /**
     * Get point with given x and y indexes.
     * @param c0_index index of c0, from 0 to c0.size()-1
     * @param c1_index index of c1, from 0 to c1.size()-1
     * @return point with given c0 and c1 indexes
     */
    Vec<2,double> operator()(std::size_t c0_index, std::size_t c1_index) const {
        return Vec<2, double>(c0[c0_index], c1[c1_index]);
    }

    /**
     * Remove all points from mesh.
     */
    void clear() {
        c0.clear();
        c1.clear();
    }

    /**
     * Calculate (using linear interpolation) value of data in point using data in points describe by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @a point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, const Vec<2, double>& point) -> typename std::remove_reference<decltype(data[0])>::type;
};

/**
 * Do linear 2d interpolation with checking bounds variants.
 * @param data 2d data source, data(i0, i1) should return data in point (c0[i0], c1[i1])
 * @param point_c0, point_c1 requested point
 * @param c0 first coordinates of points 
 * @param c1 second coordinates of points
 * @param index0 should be equal to c0.findIndex(point_c0)
 * @param index1 should be equal to c1.findIndex(point_c1)
 * @return value in point point_c0, point_c1
 * @tparam DataGetter2d functor
 */
template <typename DataGetter2d>
auto interpolateLinear2d(DataGetter2d data, const double& point_c0, const double& point_c1, const RectilinearMesh1d& c0, const RectilinearMesh1d& c1, std::size_t index0, std::size_t index1) -> typename std::remove_reference<decltype(data(0, 0))>::type {
    if (index0 == 0) {
        if (index1 == 0) return data(0, 0);
        if (index1 == c1.size()) return data(0, index1-1);
        return interpolation::linear(c1[index1-1], data(0, index1-1), c1[index1], data(0, index1), point_c1);
    }

    if (index0 == c0.size()) {
        --index0;
        if (index1 == 0) return data(index0, 0);
        if (index1 == c1.size()) return data(index0, index1-1);
        return interpolation::linear(c1[index1-1], data(index0, index1-1), c1[index1], data(index0, index1), point_c1);
    }

    if (index1 == 0)
        return interpolation::linear(c0[index0-1], data(index0-1, 0), c0[index0], data(index0, 0), point_c0);

    if (index1 == c1.size()) {
        --index1;
        return interpolation::linear(c0[index0-1], data(index0-1, index1), c0[index0], data(index0, index1), point_c0);
    }

    return interpolation::bilinear(c0[index0-1], c0[index0],
                                   c1[index1-1], c1[index1],
                                   data(index0-1, index1-1),
                                   data(index0,   index1-1),
                                   data(index0,   index1  ),
                                   data(index0-1, index1  ),
                                   point_c0, point_c1);
}


//RectilinearMesh2d method templates implementation
template <typename RandomAccessContainer>
auto RectilinearMesh2d::interpolateLinear(const RandomAccessContainer& data, const Vec<2, double>& point) -> typename std::remove_reference<decltype(data[0])>::type {    
    return interpolateLinear2d(
        [&] (std::size_t i0, std::size_t i1) { return data[index(i0, i1)]; },
        point.c0, point.c1, c0, c1, c0.findIndex(point.c0), c1.findIndex(point.c1)
    );
}

/**
 * Rectilinear mesh in 3d space.
 *
 * Includes three 1d rectilinear meshes:
 * - c0 (alternative names: lon(), ee_z(), r())
 * - c1 (alternative names: tran(), ee_x(), phi())
 * - c2 (alternative names: up(), ee_y(), z())
 * Represent all points (x, y, z) such that x is in c0, y is in c1, z is in c2.
 */
struct RectilinearMesh3d {

    ///First coordinate of points in this mesh.
    RectilinearMesh1d c0;

    ///Second coordinate of points in this mesh.
    RectilinearMesh1d c1;
    
    ///Third coordinate of points in this mesh.
    RectilinearMesh1d c2;

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    RectilinearMesh1d& lon() { return c0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const RectilinearMesh1d& lon() const { return c0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    RectilinearMesh1d& tran() { return c1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const RectilinearMesh1d& tran() const { return c1; }
    
    /**
     * Get third coordinate of points in this mesh.
     * @return c2
     */
    RectilinearMesh1d& up() { return c2; }

    /**
     * Get third coordinate of points in this mesh.
     * @return c2
     */
    const RectilinearMesh1d& up() const { return c2; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    RectilinearMesh1d& ee_z() { return c0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const RectilinearMesh1d& ee_z() const { return c0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    RectilinearMesh1d& ee_x() { return c1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const RectilinearMesh1d& ee_x() const { return c1; }
    
    /**
     * Get third coordinate of points in this mesh.
     * @return c2
     */
    RectilinearMesh1d& ee_y() { return c1; }

    /**
     * Get third coordinate of points in this mesh.
     * @return c2
     */
    const RectilinearMesh1d& ee_y() const { return c1; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    RectilinearMesh1d& rad_r() { return c0; }

    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const RectilinearMesh1d& rad_r() const { return c0; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    RectilinearMesh1d& rad_phi() { return c1; }

    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const RectilinearMesh1d& rad_phi() const { return c1; }
    
    /**
     * Get thirs coordinate of points in this mesh.
     * @return c1
     */
    RectilinearMesh1d& rad_z() { return c1; }

    /**
     * Get thirs coordinate of points in this mesh.
     * @return c1
     */
    const RectilinearMesh1d& rad_z() const { return c1; }

    ///Type of points in this mesh.
    typedef Vec<3, double> PointType;

    /**
     * Random access iterator type which allow iterate over all points in this mesh, in order appointed by operator[].
     * This iterator type is indexed, which mean that it have (read-write) index field equal to 0 for begin() and growing up to size() for end().
     */
    typedef IndexedIterator< const RectilinearMesh3d, PointType > const_iterator;

    ///@return iterator referring to the first point in this mesh
    const_iterator begin() const { return const_iterator(this, 0); }

    ///@return iterator referring to the past-the-end point in this mesh
    const_iterator end() const { return const_iterator(this, size()); }

    /**
     * Get number of points in mesh.
     * @return number of points in mesh
     */
    std::size_t size() const { return c0.size() * c1.size() * c2.size(); }

    ///@return true only if there are no points in mesh
    bool empty() const { return c0.empty() || c1.empty() || c2.empty(); }

    /**
     * Calculate this mesh index using indexes of c0, c1 and c2.
     * @param c0_index index of c0, from 0 to c0.size()-1
     * @param c1_index index of c1, from 0 to c1.size()-1
     * @param c2_index index of c2, from 0 to c2.size()-1
     * @return this mesh index, from 0 to size()-1
     */
    std::size_t index(std::size_t c0_index, std::size_t c1_index, std::size_t c2_index) const {
        return c0_index + c0.size() * (c1_index + c1.size() * c2_index);
    }

    /**
     * Calculate index of c0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of c0, from 0 to c0.size()-1
     */
    std::size_t index0(std::size_t mesh_index) const {
        return mesh_index % c0.size();
    }

    /**
     * Calculate index of c1 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of c1, from 0 to c1.size()-1
     */
    std::size_t index1(std::size_t mesh_index) const {
        return (mesh_index / c0.size()) % c1.size();
    }
    
    /**
     * Calculate index of c2 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of c2, from 0 to c2.size()-1
     */
    std::size_t index2(std::size_t mesh_index) const {
        return mesh_index / c0.size() / c1.size();
    }

    /**
     * Get point with given mesh index.
     * Points are in order: (c0[0], c1[0], c2[0]), (c0[1], c1[0], c2[0]), ..., (c0[c0.size-1], c1[0], c2[0]), (c0[0], c1[1], c2[0]), ..., (c0[c0.size()-1], c1[c1.size()-1], c2[c2.size()-1])
     * @param index index of point, from 0 to size()-1
     * @return point with given @a index
     */
    Vec<3, double> operator[](std::size_t index) const {
        const std::size_t i0 = index0(index);
        index /= c0.size();
        return operator() (i0, index % c1.size(), index / c1.size());
    }

    /**
     * Get point with given x and y indexes.
     * @param c0_index index of c0, from 0 to c0.size()-1
     * @param c1_index index of c1, from 0 to c1.size()-1
     * @param c2_index index of c2, from 0 to c2.size()-1
     * @return point with given c0, c1 and c2 indexes
     */
    Vec<3, double> operator()(std::size_t c0_index, std::size_t c1_index, std::size_t c2_index) const {
        return Vec<3, double>(c0[c0_index], c1[c1_index], c2[c2_index]);
    }

    /**
     * Remove all points from mesh.
     */
    void clear() {
        c0.clear();
        c1.clear();
        c2.clear();
    }

    /**
     * Calculate (using linear interpolation) value of data in point using data in points describe by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @a point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, const Vec<3, double>& point) -> typename std::remove_reference<decltype(data[0])>::type;
};

//RectilinearMesh3d method templates implementation
template <typename RandomAccessContainer>
auto RectilinearMesh3d::interpolateLinear(const RandomAccessContainer& data, const Vec<3, double>& point) -> typename std::remove_reference<decltype(data[0])>::type {
    std::size_t index0 = c0.findIndex(point.c0);
    std::size_t index1 = c1.findIndex(point.c1);
    std::size_t index2 = c2.findIndex(point.c2);
    
    if (index2 == 0)
        return interpolateLinear2d(
            [&] (std::size_t i0, std::size_t i1) { return data[index(i0, i1, 0)]; },
            point.c0, point.c1, c0, c1, index0, index1
        );
    
    if (index2 == c0.size()) {
        --index2;
        return interpolateLinear2d(
            [&] (std::size_t i0, std::size_t i1) { return data[index(i0, i1, index2)]; },
            point.c0, point.c1, c0, c1, index0, index1
        );
    }
    
    if (index1 == 0)
        return interpolateLinear2d(
            [&] (std::size_t i0, std::size_t i2) { return data[index(i0, 0, i2)]; },
            point.c0, point.c2, c0, c2, index0, index2
        );
    
    if (index1 == c1.size()) {
        --index1;
        return interpolateLinear2d(
            [&] (std::size_t i0, std::size_t i2) { return data[index(i0, index1, i2)]; },
            point.c0, point.c2, c0, c2, index0, index2
        );
    }
    
    //index1 and index2 are in bounds here:
    if (index0 == 0)
       return interpolation::bilinear(c1[index1-1], c1[index1],
                                      c2[index2-1], c2[index2],
                                      data[index(0, index1-1, index2-1)],
                                      data[index(0, index1,   index2-1)],
                                      data[index(0, index1,   index2  )],
                                      data[index(0, index1-1, index2  )],
                                      point.c1, point.c2);
    if (index0 == c0.size()) {
        --index0;
       return interpolation::bilinear(c1[index1-1], c1[index1],
                                      c2[index2-1], c2[index2],
                                      data[index(index0, index1-1, index2-1)],
                                      data[index(index0, index1,   index2-1)],
                                      data[index(index0, index1,   index2  )],
                                      data[index(index0, index1-1, index2  )],
                                      point.c1, point.c2);
    }
    
    //all indexes are in bounds
    return interpolation::trilinear(
        c0[index0-1], c0[index0],
        c1[index1-1], c1[index1],
        c2[index2-1], c2[index2],
        data[index(index0-1, index1-1, index2-1)],
        data[index(index0,   index1-1, index2-1)],
        data[index(index0,   index1  , index2-1)],
        data[index(index0-1, index1  , index2-1)],
        data[index(index0-1, index1-1, index2)],
        data[index(index0,   index1-1, index2)],
        data[index(index0,   index1  , index2)],
        data[index(index0-1, index1  , index2)],
        point.c0, point.c1, point.c2
    );
    
}

}	//namespace plask

#endif // PLASK__RECTILINEAR_H
