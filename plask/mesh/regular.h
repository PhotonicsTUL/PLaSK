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


#endif // PLASK__REGULAR_H
