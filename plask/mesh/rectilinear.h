#ifndef PLASK__RECTILINEAR_H
#define PLASK__RECTILINEAR_H

/** @file
This file includes rectilinear meshes for 1d, 2d, and 3d spaces.
*/

#include <vector>
#include <algorithm>
#include <initializer_list>

#include "../vector/2d.h"
#include "../utils/iterators.h"

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

    ///Random access iterator type which alow iterate over all points in this mesh, in ascending order.
    typedef std::vector<double>::const_iterator const_iterator;

    ///@return iterator referring to the first point in this mesh
    const_iterator begin() const { return points.begin(); }

    ///@return iterator referring to the past-the-end point in this mesh
    const_iterator end() const { return points.end(); }

    /**
     * Find position where @a to_find point could be insert.
     * @param to_find point to find
     * @return First position where to_find could be insert.
     *         Reffer to value equal to @a to_find only if @a to_find is already in mesh.
     *         Can be equal to end() if to_find is higher than all points in mesh
     *         (in such case returned iterator can't be dereferenced).
     */
    const_iterator find(double to_find) const;

    /**
     * Find index where @a to_find point could be insert.
     * @param to_find point to find
     * @return First index where to_find could be insert.
     *         Reffer to value equal to @a to_find only if @a to_find is already in mesh.
     *         Can be equal to size() if to_find is higher than all points in mesh.
     */
    std::size_t findIndex(double to_find) const { return find(to_find) - begin(); }

    /**
     * Calculate (using linear interpolation) value of data in point using data in points describe by this mesh.
     * @param data values of data in points describe by this mesh
     * @param point point in which value should be calculate
     * @return interpolated value in point @a point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(RandomAccessContainer& data, double point) -> typename std::remove_reference<decltype(data[0])>::type;

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
     * @return true only if this mesh and to_compare represents the same set of points
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
     * It use algorithm which has linear time complexity.
     * @param begin, end ordered range of points in ascending order
     * @tparam RandomAccessIteratorT random access iterator
     */
    //TODO use iterator traits and write version for input iterator
    template <typename RandomAccessIteratorT>
    void addOrderedPoints(RandomAccessIteratorT begin, RandomAccessIteratorT end) { addOrderedPoints(begin, end, end - begin); }
    
    /**
     * Add to mesh points: first + i * len / points_count, where i is in range [0, points_count].
     * It use algorithm which has linear time complexity.
     * @param first coordinate of first point
     * @param len first+len is coordinate of last point
     * @param points_count number of points to add
     */
    void addPoints(double first, double len, std::size_t points_count);

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

};

//RectilinearMesh1d method templates implementation
template <typename RandomAccessContainer>
auto RectilinearMesh1d::interpolateLinear(RandomAccessContainer& data, double point) -> typename std::remove_reference<decltype(data[0])>::type {
    std::size_t index = findIndex(point);
    if (index == size()) return data[index - 1];     //TODO what should do if mesh is empty?
    if (index == 0 || operator [](index) == point) return data[index]; //hit exactly
    //here: d0=data[index-1] < point < data[index]=d1
    //TODO which one is more stable?
    //auto d0 = data[index-1];
    //return d0 + (data[index] - d0) * (point - operator[](index-1)) / (operator[](index) - operator[](index-1));
    return ((operator[](index) - point) * data[index-1] + (point - operator[](index-1)) * data[index])
                        / (operator[](index) - operator[](index-1));
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
 * - c0 (alternative names: x(), r())
 * - c1 (alternative names: y(), z())
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

        typedef Vec2<double> PointType;

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
        Vec2<double> operator[](std::size_t index) const {
            if (changed) {
                const std::size_t y_size = mesh.c1.size();
                return Vec2<double>(mesh.c0[index / y_size], mesh.c1[index % y_size]);
            } else {
                return mesh[index];
            }
        }

        std::size_t size() const { return mesh.size(); }

        std::size_t index(std::size_t x_index, std::size_t y_index) const {
            return changed ? mesh.c1.size() * x_index + y_index : mesh.index(x_index, y_index);
        }

        /**
        * Calculate index of c0 using this mesh index.
        * @param mesh_index this mesh index, from 0 to size()-1
        * @return index of c0, from 0 to c0.size()-1
        */
        std::size_t index0(std::size_t mesh_index) const {
            return changed ? mesh_index / mesh.c1.size() : index0(mesh_index);
        }

        /**
        * Calculate index of c1 using this mesh index.
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
    RectilinearMesh1d& x() { return c0; }
    
    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const RectilinearMesh1d& x() const { return c0; }
    
    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    RectilinearMesh1d& y() { return c1; }
    
    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const RectilinearMesh1d& y() const { return c1; }
    
    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    RectilinearMesh1d& r() { return c0; }
    
    /**
     * Get first coordinate of points in this mesh.
     * @return c0
     */
    const RectilinearMesh1d& r() const { return c0; }
    
    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    RectilinearMesh1d& z() { return c1; }
    
    /**
     * Get second coordinate of points in this mesh.
     * @return c1
     */
    const RectilinearMesh1d& z() const { return c1; }

    ///Type of points in this mesh.
    typedef Vec2<double> PointType;

    /**
     * Random access iterator type which alow iterate over all points in this mesh, in order appointed by operator[].
     * This iterator type is indexed, which mean that it have (read-write) index field equal to 0 for begin() and growing up to size() for end().
     */
    typedef IndexedIterator< const RectilinearMesh2d, PointType > const_iterator;

    ///@return iterator referring to the first point in this mesh
    const_iterator begin() const { return const_iterator(this, 0); }

    ///@return iterator referring to the past-the-end point in this mesh
    const_iterator end() const { return const_iterator(this, size()); }

    ///@return number of points in mesh
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
    Vec2<double> operator[](std::size_t index) const {
        const std::size_t c0_size = c0.size();
        return Vec2<double>(c0[index % c0_size], c1[index / c0_size]);
    }

    /**
     * Get point with given x and y indexes.
     * @param c0_index index of c0, from 0 to c0.size()-1
     * @param c1_index index of c1, from 0 to c1.size()-1
     * @return point with given c0 and c1 indexes
     */
    Vec2<double> operator()(std::size_t c0_index, std::size_t c1_index) const {
        return Vec2<double>(c0[c0_index], c1[c1_index]);
    }
    
    /**
     * Remove all points from mesh.
     */
    void clear() {
        c0.clear();
        c1.clear();
    }
};

}	//namespace plask

#endif // PLASK__RECTILINEAR_H
