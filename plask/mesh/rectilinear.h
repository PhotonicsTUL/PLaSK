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

#endif // PLASK__RECTILINEAR_H
