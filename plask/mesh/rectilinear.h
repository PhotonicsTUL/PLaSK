#ifndef PLASK__RECTILINEAR_H
#define PLASK__RECTILINEAR_H

#include <vector>

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

    //Construct empty mesh.
    //RectilinearMesh1d() {}

    ///@return number of points in mesh
    std::size_t size() const { return points.size(); }

    /**
     * Add (1d) point to this mesh.
     * Point is add to mesh only if it is not already included in mesh.
     * @param new_node_cord coordinate of point to add
     */
    void addPoint(double new_node_cord);

    /**
     * Get point by index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @a index
     */
    const double& operator[](std::size_t index) const { return points[index]; }

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

/**
 * Rectilinear mesh in 2d space.
 */
struct RectilinearMesh2d {

    /**
     * Class which allow to access and iterate over RectilinearMesh2d points in choosen order.
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
        * - if its set: (c0[0], y[0]), (c0[0], y[1]), ..., (c0[0], y[y.size-1]), (c0[1], y[0]), ..., (c0[c0.size()-1], y[y.size()-1])
        * - if its not set: (c0[0], y[0]), (c0[1], y[0]), ..., (c0[c0.size-1], y[0]), (c0[0], y[1]), ..., (c0[c0.size()-1], y[y.size()-1])
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
        * Calculate index of x using this mesh index.
        * @param mesh_index this mesh index, from 0 to size()-1
        * @return index of x, from 0 to x.size()-1
        */
        std::size_t index0(std::size_t mesh_index) const {
            return changed ? mesh_index / mesh.c1.size() : index0(mesh_index);
        }

        /**
        * Calculate index of y using this mesh index.
        * @param mesh_index this mesh index, from 0 to size()-1
        * @return index of y, from 0 to y.size()-1
        */
        std::size_t index1(std::size_t mesh_index) const {
            return changed ? mesh_index % mesh.c1.size() : index1(mesh_index);
        }

    };

    ///First coordinate of points in this mesh.
    RectilinearMesh1d c0;

    ///Second coordinate of points in this mesh.
    RectilinearMesh1d c1;

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

    /**
     * Calculate this mesh index using indexes of x and y.
     * @param x_index index of x, from 0 to x.size()-1
     * @param y_index index of y, from 0 to y.size()-1
     * @return this mesh index, from 0 to size()-1
     */
    std::size_t index(std::size_t c0_index, std::size_t y_index) const {
        return c0_index + c0.size() * y_index;
    }

    /**
     * Calculate index of x using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of x, from 0 to x.size()-1
     */
    std::size_t index0(std::size_t mesh_index) const {
        return mesh_index % c0.size();
    }

    /**
     * Calculate index of y using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of y, from 0 to y.size()-1
     */
    std::size_t index1(std::size_t mesh_index) const {
        return mesh_index / c0.size();
    }

    /**
     * Get point with given mesh index.
     * Points are in order: (x[0], y[0]), (x[1], y[0]), ..., (x[x.size-1], y[0]), (x[0], y[1]), ..., (x[x.size()-1], y[y.size()-1])
     * @param index index of point, from 0 to size()-1
     * @return point with given @a index
     */
    Vec2<double> operator[](std::size_t index) const {
        const std::size_t c0_size = c0.size();
        return Vec2<double>(c0[index % c0_size], c1[index / c0_size]);
    }

    /**
     * Get point with given x and y indexes.
     * @param x_index index of x, from 0 to x.size()-1
     * @param y_index index of y, from 0 to y.size()-1
     * @return point with given x and y indexes
     */
    Vec2<double> operator()(std::size_t c0_index, std::size_t c1_index) const {
        return Vec2<double>(c0[c0_index], c1[c1_index]);
    }
};

}	//namespace plask

#endif // PLASK__RECTILINEAR_H
