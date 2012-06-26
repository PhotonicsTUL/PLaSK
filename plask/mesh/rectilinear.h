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
class RectilinearMesh2d: public RectangularMesh2d<RectilinearMesh1d> {

    void buildFromGeometry(const GeometryElementD<2>& geometry);

  public:

    /// Construct an empty mesh
    RectilinearMesh2d(RectangularMesh2d<RectilinearMesh1d>::IterationOrder iterationOrder = RectangularMesh2d<RectilinearMesh1d>::NORMAL_ORDER) :
        RectangularMesh2d<RectilinearMesh1d>(iterationOrder) {}

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     */
    RectilinearMesh2d(const RectilinearMesh1d& mesh0, const RectilinearMesh1d& mesh1,
                      RectangularMesh2d<RectilinearMesh1d>::IterationOrder iterationOrder = RectangularMesh2d<RectilinearMesh1d>::NORMAL_ORDER) :
        RectangularMesh2d<RectilinearMesh1d>(mesh0, mesh1, iterationOrder) {}

    /**
     * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
     * @param geometry geometry in which bounding boxes are searched
     */
    RectilinearMesh2d(const GeometryElementD<2>& geometry, IterationOrder iterationOrder=RectangularMesh2d<RectilinearMesh1d>::NORMAL_ORDER) {
        buildFromGeometry(geometry);
        this->setIterationOrder(iterationOrder);
    }

    /**
     * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
     * @param geometry geometry in which bounding boxes are searched
     */
    RectilinearMesh2d(shared_ptr<const GeometryElementD<2>> geometry, IterationOrder iterationOrder=RectangularMesh2d<RectilinearMesh1d>::NORMAL_ORDER) {
        buildFromGeometry(*geometry);
        this->setIterationOrder(iterationOrder);
    }

    /**
     * Return a mesh that enables iterating over middle points of the rectangles
     * \return new rectilinear mesh with points in the middles of original rectangles
     */
    RectilinearMesh2d getMidpointsMesh() const;
};

/// Three-dimensional rectilinear mesh type
class RectilinearMesh3d: public RectangularMesh3d<RectilinearMesh1d> {

    void buildFromGeometry(const GeometryElementD<3>& geometry);

  public:

    /// Construct an empty mesh
    RectilinearMesh3d(RectangularMesh3d<RectilinearMesh1d>::IterationOrder iterationOrder = RectangularMesh3d<RectilinearMesh1d>::ORDER_021) :
        RectangularMesh3d<RectilinearMesh1d>(iterationOrder) {}

    /**
     * Construct mesh with is based on given 1D meshes
     *
     * @param mesh0 mesh for the first coordinate
     * @param mesh1 mesh for the second coordinate
     */
    RectilinearMesh3d(const RectilinearMesh1d& mesh0, const RectilinearMesh1d& mesh1, const RectilinearMesh1d& mesh2,
                      RectangularMesh3d<RectilinearMesh1d>::IterationOrder iterationOrder = RectangularMesh3d<RectilinearMesh1d>::ORDER_021) :
        RectangularMesh3d<RectilinearMesh1d>(mesh0, mesh1, mesh2, iterationOrder) {}


    /**
     * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
     * @param geometry geometry in which bounding boxes are searched
     */
    RectilinearMesh3d(const GeometryElementD<3>& geometry, IterationOrder iterationOrder = RectangularMesh3d<RectilinearMesh1d>::ORDER_021) {
        buildFromGeometry(geometry);
        this->setIterationOrder(iterationOrder);
    }

    /**
     * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
     * @param geometry geometry in which bounding boxes are searched
     */
    RectilinearMesh3d(shared_ptr<const GeometryElementD<3>> geometry, IterationOrder iterationOrder = RectangularMesh3d<RectilinearMesh1d>::ORDER_021) {
        buildFromGeometry(*geometry);
        this->setIterationOrder(iterationOrder);
    }

    /**
     * Return a mesh that enables iterating over middle points of the rectangles
     * \return new rectilinear mesh with points in the middles of original rectangles
     */
    RectilinearMesh3d getMidpointsMesh() const;
};


} // namespace  plask


#endif // PLASK__RECTILINEAR_H
