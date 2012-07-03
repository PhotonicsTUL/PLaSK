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
typedef RectangularMesh2D<RectilinearMesh1D> RectilinearMesh2D;

/// Three-dimensional rectilinear mesh type
typedef RectangularMesh3D<RectilinearMesh1D> RectilinearMesh3D;



/**
* Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
* @param geometry geometry in which bounding boxes are searched
* @param iterationOrder iteration order
*/
RectilinearMesh2D RectilinearMeshFromGeometry(const GeometryElementD<2>& geometry, RectilinearMesh2D::IterationOrder iterationOrder=RectilinearMesh2D::NORMAL_ORDER);

/**
* Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
* @param geometry geometry in which bounding boxes are searched
* @param iterationOrder iteration order
*/
inline RectilinearMesh2D RectilinearMeshFromGeometry(shared_ptr<const GeometryElementD<2>> geometry, RectilinearMesh2D::IterationOrder iterationOrder=RectilinearMesh2D::NORMAL_ORDER) {
    return RectilinearMeshFromGeometry(*geometry, iterationOrder);
}

/**
    * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
    * @param geometry geometry in which bounding boxes are searched
    * @param iterationOrder iteration order
    */
RectilinearMesh3D RectilinearMeshFromGeometry(const GeometryElementD<3>& geometry, RectilinearMesh3D::IterationOrder iterationOrder=RectilinearMesh3D::ORDER_210);

/**
    * Construct mesh with lines along boundaries of bounding boxes of all leafs in geometry
    * @param geometry geometry in which bounding boxes are searched
    * @param iterationOrder iteration order
    */
inline RectilinearMesh3D RectilinearMeshFromGeometry(shared_ptr<const GeometryElementD<3>> geometry, RectilinearMesh3D::IterationOrder iterationOrder=RectilinearMesh3D::ORDER_210)  {
    return RectilinearMeshFromGeometry(*geometry, iterationOrder);
}

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
