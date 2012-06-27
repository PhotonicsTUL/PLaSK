#ifndef PLASK__RECTILINEAR_H
#define PLASK__RECTILINEAR_H

/** @file
This file includes rectilinear meshes for 1d, 2d, and 3d spaces.
*/

#include "rectilinear1d.h"

#include "rectangular2d.h"
#include "rectangular3d.h"

namespace plask {

/// Two-dimensional rectilinear mesh type
class RectilinearMesh2D: public RectangularMesh2D<RectilinearMesh1D> {

    void buildFromGeometry(const GeometryElementD<2>& geometry);

  public:

    /// Construct an empty mesh
    RectilinearMesh2D(RectangularMesh2D<RectilinearMesh1D>::IterationOrder iterationOrder = RectangularMesh2D<RectilinearMesh1D>::NORMAL_ORDER) :
        RectangularMesh2D<RectilinearMesh1D>(iterationOrder) {}

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     */
    RectilinearMesh2D(const RectilinearMesh1D& mesh0, const RectilinearMesh1D& mesh1,
                      RectangularMesh2D<RectilinearMesh1D>::IterationOrder iterationOrder = RectangularMesh2D<RectilinearMesh1D>::NORMAL_ORDER) :
        RectangularMesh2D<RectilinearMesh1D>(mesh0, mesh1, iterationOrder) {}

    /**
     * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
     * @param geometry geometry in which bounding boxes are searched
     */
    RectilinearMesh2D(const GeometryElementD<2>& geometry, IterationOrder iterationOrder=RectangularMesh2D<RectilinearMesh1D>::NORMAL_ORDER) {
        buildFromGeometry(geometry);
        this->setIterationOrder(iterationOrder);
    }

    /**
     * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
     * @param geometry geometry in which bounding boxes are searched
     */
    RectilinearMesh2D(shared_ptr<const GeometryElementD<2>> geometry, IterationOrder iterationOrder=RectangularMesh2D<RectilinearMesh1D>::NORMAL_ORDER) {
        buildFromGeometry(*geometry);
        this->setIterationOrder(iterationOrder);
    }

    /**
     * Return a mesh that enables iterating over middle points of the rectangles
     * \return new rectilinear mesh with points in the middles of original rectangles
     */
    RectilinearMesh2D getMidpointsMesh() const;
};

/// Three-dimensional rectilinear mesh type
class RectilinearMesh3D: public RectangularMesh3D<RectilinearMesh1D> {

    void buildFromGeometry(const GeometryElementD<3>& geometry);

  public:

    /// Construct an empty mesh
    RectilinearMesh3D(RectangularMesh3D<RectilinearMesh1D>::IterationOrder iterationOrder = RectangularMesh3D<RectilinearMesh1D>::ORDER_021) :
        RectangularMesh3D<RectilinearMesh1D>(iterationOrder) {}

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     */
    RectilinearMesh3D(const RectilinearMesh1D& mesh0, const RectilinearMesh1D& mesh1, const RectilinearMesh1D& mesh2,
                      RectangularMesh3D<RectilinearMesh1D>::IterationOrder iterationOrder = RectangularMesh3D<RectilinearMesh1D>::ORDER_021) :
        RectangularMesh3D<RectilinearMesh1D>(mesh0, mesh1, mesh2, iterationOrder) {}


    /**
     * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
     * @param geometry geometry in which bounding boxes are searched
     */
    RectilinearMesh3D(const GeometryElementD<3>& geometry, IterationOrder iterationOrder = RectangularMesh3D<RectilinearMesh1D>::ORDER_021) {
        buildFromGeometry(geometry);
        this->setIterationOrder(iterationOrder);
    }

    /**
     * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
     * @param geometry geometry in which bounding boxes are searched
     */
    RectilinearMesh3D(shared_ptr<const GeometryElementD<3>> geometry, IterationOrder iterationOrder = RectangularMesh3D<RectilinearMesh1D>::ORDER_021) {
        buildFromGeometry(*geometry);
        this->setIterationOrder(iterationOrder);
    }

    /**
     * Return a mesh that enables iterating over middle points of the rectangles
     * \return new rectilinear mesh with points in the middles of original rectangles
     */
    RectilinearMesh3D getMidpointsMesh() const;
};


} // namespace  plask

namespace std { // use fast iterator if we know mesh type at compile time:

    inline auto begin(const plask::RectilinearMesh2D& m) -> decltype(m.begin_fast()) {
        return m.begin_fast();
    }

    inline auto end(const plask::RectilinearMesh2D& m) -> decltype(m.end_fast()) {
        return m.end_fast();
    }

    inline auto begin(const plask::RectilinearMesh3D& m) -> decltype(m.begin_fast()) {
        return m.begin_fast();
    }

    inline auto end(const plask::RectilinearMesh3D& m) -> decltype(m.end_fast()) {
        return m.end_fast();
    }

} // namespace std



#endif // PLASK__RECTILINEAR_H
