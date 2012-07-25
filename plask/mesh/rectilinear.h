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
typedef RectangularMesh<2,RectilinearMesh1D> RectilinearMesh2D;

/// Three-dimensional rectilinear mesh type
typedef RectangularMesh<3,RectilinearMesh1D> RectilinearMesh3D;

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
