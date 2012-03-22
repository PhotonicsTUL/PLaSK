#ifndef PLASK__RECTILINEAR3D_H
#define PLASK__RECTILINEAR3D_H

/** @file
This file includes rectilinear mesh for 3d space.
*/

#include "rectilinear2d.h"

namespace plask {

/**
 * Rectilinear mesh in 3d space.
 *
 * Includes three 1d rectilinear meshes:
 * - c0 (alternative names: lon(), ee_z(), r())
 * - c1 (alternative names: tran(), ee_x(), phi())
 * - c2 (alternative names: up(), ee_y(), z())
 * Represent all points (x, y, z) such that x is in c0, y is in c1, z is in c2.
 */
struct RectilinearMesh3d: public Mesh<3> {

    /// First coordinate of points in this mesh.
    RectilinearMesh1d c0;

    /// Second coordinate of points in this mesh.
    RectilinearMesh1d c1;

    /// Third coordinate of points in this mesh.
    RectilinearMesh1d c2;

    /// Construct an empty mesh
    RectilinearMesh3d() {}

    /**
     * Construct mesh with based on gived 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param mesh2 mesh for the third coordinate
     */
    RectilinearMesh3d(const RectilinearMesh1d& mesh0, const RectilinearMesh1d& mesh1, const RectilinearMesh1d& mesh2) :
        c0(mesh0), c1(mesh1), c2(mesh2) {}

    /**
     * Construct mesh with given points.
     * It uses algorithm which has quadric time complexity.
     *
     * @param points0 points along the first coordinate (in any order)
     * @param points1 points along the second coordinate (in any order)
     * @param points1 points along the third coordinate (in any order)
     */
    RectilinearMesh3d(std::initializer_list<RectilinearMesh1d::PointType> points0,
                      std::initializer_list<RectilinearMesh1d::PointType> points1,
                      std::initializer_list<RectilinearMesh1d::PointType> points2) :
        c0(points0), c1(points1), c2(points2) {}

    /**
     * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
     *
     * @param geometry geometry in which bounding boxes are searched
     */
    RectilinearMesh3d(const GeometryElementD<3>& geometry) {
        buildFromGeometry(geometry);
    }

    /**
     * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
     *
     * @param geometry geometry in which bounding boxes are searched
     */
    RectilinearMesh3d(shared_ptr<const GeometryElementD<3>> geometry) {
        buildFromGeometry(*geometry);
    }


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
    const_iterator begin_fast() const { return const_iterator(this, 0); }

    ///@return iterator referring to the past-the-end point in this mesh
    const_iterator end_fast() const { return const_iterator(this, size()); }

    //implement Mesh<3> polimorphic iterators:
    virtual typename Mesh<3>::Iterator begin() const { return makeMeshIterator(begin_fast()); }
    virtual typename Mesh<3>::Iterator end() const { return makeMeshIterator(end_fast()); }

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
     * @return point with given @p index
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
     * @return interpolated value in point @p point
     */
    template <typename RandomAccessContainer>
    auto interpolateLinear(const RandomAccessContainer& data, const Vec<3, double>& point) -> typename std::remove_reference<decltype(data[0])>::type;


  private:

    void buildFromGeometry(const GeometryElementD<3>& geometry);

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

template <typename DataT>    //for any data type
struct InterpolationAlgorithm<RectilinearMesh3d, DataT, LINEAR> {
    static void interpolate(RectilinearMesh3d& src_mesh, const std::vector<DataT>& src_vec, const plask::Mesh<3>& dst_mesh, std::vector<DataT>& dst_vec) {
        for (auto p: dst_mesh)
            dst_vec.push_back(src_mesh.interpolateLinear(dst_vec, p));
    }
};

}   // namespace plask

namespace std { //use fast iterator if know mesh type at compile time:

    inline auto begin(const plask::RectilinearMesh3d& m) -> decltype(m.begin_fast()) {
      return m.begin_fast();
    }

    inline auto end(const plask::RectilinearMesh3d& m) -> decltype(m.end_fast()) {
      return m.begin_fast();
    }

}   // namespace std

#endif // PLASK__RECTILINEAR3D_H
