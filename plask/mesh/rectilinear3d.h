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
//TODO methods which call fireResize() when points are adding, etc.
class RectilinearMesh3d: public Mesh<3> {

    typedef std::size_t index_ft(const RectilinearMesh3d* mesh, std::size_t c0_index, std::size_t c1_index, std::size_t c2_index);
    typedef std::size_t index012_ft(const RectilinearMesh3d* mesh, std::size_t mesh_index);

    // our own virtual table, changeable in run-time:
    index_ft* index_f;
    index012_ft* index0_f;
    index012_ft* index1_f;
    index012_ft* index2_f;

  public:

    /// First coordinate of points in this mesh.
    RectilinearMesh1d c0;

    /// Second coordinate of points in this mesh.
    RectilinearMesh1d c1;

    /// Third coordinate of points in this mesh.
    RectilinearMesh1d c2;

    /**
     * Iteration orders:
     * - normal iteration order (ORDER_012) is:
     *   (c0[0], c1[0]), (c0[1], c1[0]), ..., (c0[c0.size-1], c1[0]), (c0[0], c1[1]), ..., (c0[c0.size()-1], c1[c1.size()-1])
     * Every other order is proper permutation of indices
     * @see setIterationOrder, getIterationOrder, setOptimalIterationOrder
     */
    enum IterationOrder { ORDER_012, ORDER_021, ORDER_102, ORDER_120, ORDER_201, ORDER_210 };

    /// Construct an empty mesh
    RectilinearMesh3d(IterationOrder iterationOrder=ORDER_210) {
        setIterationOrder(iterationOrder);
    }

    /**
     * Choose iteration order.
     * @param order iteration order to use
     * @see IterationOrder
     */
    void setIterationOrder(IterationOrder order);

    /**
     * Get iteration order.
     * @return iteration order used by this mesh
     * @see IterationOrder
     */
    IterationOrder getIterationOrder() const;

    /**
     * Set iteration order to the shortest axis changes fastest.
     */
    void setOptimalIterationOrder();

    /**
     * Construct mesh with based on gived 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     * @param mesh2 mesh for the third coordinate
     */
    RectilinearMesh3d(const RectilinearMesh1d& mesh0, const RectilinearMesh1d& mesh1, const RectilinearMesh1d& mesh2, IterationOrder iterationOrder=ORDER_012)
        : c0(mesh0), c1(mesh1), c2(mesh2) {
            setIterationOrder(iterationOrder);
    }

    /**
     * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
     *
     * @param geometry geometry in which bounding boxes are searched
     */
    RectilinearMesh3d(const GeometryElementD<3>& geometry, IterationOrder iterationOrder=ORDER_012) {
        buildFromGeometry(geometry);
        setIterationOrder(iterationOrder);
    }

    /**
     * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
     *
     * @param geometry geometry in which bounding boxes are searched
     */
    RectilinearMesh3d(shared_ptr<const GeometryElementD<3>> geometry, IterationOrder iterationOrder=ORDER_012) {
        buildFromGeometry(*geometry);
        setIterationOrder(iterationOrder);
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

    // implement Mesh<3> polymorphic iterators:
    virtual typename Mesh<3>::Iterator begin() const { return makeMeshIterator(begin_fast()); }
    virtual typename Mesh<3>::Iterator end() const { return makeMeshIterator(end_fast()); }

    /**
     * Get number of points in the mesh.
     * @return number of points in the mesh
     */
    virtual std::size_t size() const { return c0.size() * c1.size() * c2.size(); }

    /// @return true only if there are no points in mesh
    bool empty() const { return c0.empty() || c1.empty() || c2.empty(); }

    /**
     * Calculate this mesh index using indexes of c0, c1 and c2.
     * @param c0_index index of c0, from 0 to c0.size()-1
     * @param c1_index index of c1, from 0 to c1.size()-1
     * @param c2_index index of c2, from 0 to c2.size()-1
     * @return this mesh index, from 0 to size()-1
     */
    std::size_t index(std::size_t c0_index, std::size_t c1_index, std::size_t c2_index) const {
        return index_f(this, c0_index, c1_index, c2_index);
    }

    /**
     * Calculate index of c0 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of c0, from 0 to c0.size()-1
     */
    inline std::size_t index0(std::size_t mesh_index) const {
       return index0_f(this, mesh_index);
    }

    /**
     * Calculate index of c1 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of c1, from 0 to c1.size()-1
     */
    inline std::size_t index1(std::size_t mesh_index) const {
        return index1_f(this, mesh_index);
    }

    /**
     * Calculate index of c2 using this mesh index.
     * @param mesh_index this mesh index, from 0 to size()-1
     * @return index of c2, from 0 to c2.size()-1
     */
    inline std::size_t index2(std::size_t mesh_index) const {
        return index2_f(this, mesh_index);
    }

    /**
     * Get point with given mesh index.
     * Points are in order: (c0[0], c1[0], c2[0]), (c0[1], c1[0], c2[0]), ..., (c0[c0.size-1], c1[0], c2[0]), (c0[0], c1[1], c2[0]), ..., (c0[c0.size()-1], c1[c1.size()-1], c2[c2.size()-1])
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    inline Vec<3, double> operator[](std::size_t index) const {
        return operator() (index0(index), index1(index), index2(index));
    }

    /**
     * Get point with given x and y indexes.
     * @param c0_index index of c0, from 0 to c0.size()-1
     * @param c1_index index of c1, from 0 to c1.size()-1
     * @param c2_index index of c2, from 0 to c2.size()-1
     * @return point with given c0, c1 and c2 indexes
     */
    inline Vec<3, double> operator()(std::size_t c0_index, std::size_t c1_index, std::size_t c2_index) const {
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
     * Return a mesh that enables iterating over middle points of the rectangles
     * \return new rectilinear mesh with points in the middles of original rectangles
     */
    RectilinearMesh3d getMidpointsMesh() const;

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

// RectilinearMesh3d method templates implementation
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
struct InterpolationAlgorithm<RectilinearMesh3d, DataT, INTERPOLATION_LINEAR> {
    static void interpolate(RectilinearMesh3d& src_mesh, const DataVector<DataT>& src_vec, const plask::Mesh<3>& dst_mesh, DataVector<DataT>& dst_vec) {
        auto dst = dst_vec.begin();
        for (auto p: dst_mesh)
            *dst++ = src_mesh.interpolateLinear(src_vec, p);
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
