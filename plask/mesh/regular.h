#ifndef PLASK__REGULAR_H
#define PLASK__REGULAR_H

/** @file
This file includes rectilinear meshes for 1d, 2d, and 3d spaces.
*/

#include "regular1d.h"

#include "rectangular2d.h"
#include "rectangular3d.h"

namespace plask {

/// Two-dimensional rectilinear mesh type
typedef RectangularMesh<2,RegularMesh1D> RegularMesh2D;

/// Three-dimensional rectilinear mesh type
typedef RectangularMesh<3,RegularMesh1D> RegularMesh3D;

} // namespace plask


namespace std { // use fast iterator if we know mesh type at compile time:

    inline auto begin(const plask::RegularMesh2D& m) -> decltype(m.begin_fast()) {
        return m.begin_fast();
    }

    inline auto end(const plask::RegularMesh2D& m) -> decltype(m.end_fast()) {
        return m.end_fast();
    }

    inline auto begin(const plask::RegularMesh3D& m) -> decltype(m.begin_fast()) {
        return m.begin_fast();
    }

    inline auto end(const plask::RegularMesh3D& m) -> decltype(m.end_fast()) {
        return m.end_fast();
    }

} // namespace std


#endif // PLASK__REGULAR_H
