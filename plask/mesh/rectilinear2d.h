#ifndef PLASK__RECTILINEAR2D_H
#define PLASK__RECTILINEAR2D_H

/** @file
This file includes rectilinear mesh for 2d space.
*/

#include "rectilinear1d.h"
#include "mesh.h"
#include "interpolation.h"

namespace plask {

/**
 * Rectilinear mesh in 2d space.
 *
 * Includes two 1d rectilinear meshes:
 * - c0 (alternative names: tran(), ee_x(), r())
 * - c1 (alternative names: up(), ee_y(), z())
 * Represent all points (x, y) such that x is in c0 and y is in c1.
 */
struct RectilinearMesh2d {

    typedef SimpleMeshAdapter<RectilinearMesh2d, space::Cartesian2d> ExternalCartesian;
    typedef SimpleMeshAdapter<RectilinearMesh2d, space::Cylindrical2d> ExternalCylindrical;

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
        * @return point with given @p index
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
     * @return point with given @p index
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
     * @return interpolated value in point @p point
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

template <typename DataT>    //for any data type
struct InterpolationAlgorithm<RectilinearMesh2d::ExternalCartesian, DataT, LINEAR> {
    static void interpolate(RectilinearMesh2d::ExternalCartesian& src_mesh, const std::vector<DataT>& src_vec, const plask::Mesh<typename RectilinearMesh2d::ExternalCartesian::Space>& dst_mesh, std::vector<DataT>& dst_vec) {
        for (auto p: dst_mesh)
            dst_vec.push_back(src_mesh.internal.interpolateLinear(dst_vec, p));
    }
};

template <typename DataT>    //for any data type
struct InterpolationAlgorithm<RectilinearMesh2d::ExternalCylindrical, DataT, LINEAR> {
    static void interpolate(RectilinearMesh2d::ExternalCylindrical& src_mesh, const std::vector<DataT>& src_vec, const plask::Mesh<typename RectilinearMesh2d::ExternalCylindrical::Space>& dst_mesh, std::vector<DataT>& dst_vec) {
        for (auto p: dst_mesh)
            dst_vec.push_back(src_mesh.internal.interpolateLinear(dst_vec, p));
    }
};

}   // namespace plask

#endif // PLASK__RECTILINEAR2D_H
