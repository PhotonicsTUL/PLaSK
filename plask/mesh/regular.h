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
typename RectangularMesh2d<RegularMesh1d> RegularMesh2d;

/// Three-dimensional rectilinear mesh type
typename RectangularMesh2d<RegularMesh1d> RegularMesh3d;



} // namespace plask

#endif // PLASK__REGULAR_H