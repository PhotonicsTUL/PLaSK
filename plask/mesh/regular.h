#ifndef PLASK__REGULAR_H
#define PLASK__REGULAR_H

/** @file
This file contains rectilinear meshes for 1d, 2d, and 3d spaces.
*/

#include "regular1d.h"

#include "rectangular2d.h"
#include "rectangular3d.h"

namespace plask {

/// Two-dimensional rectilinear mesh type
typedef RectangularMesh<2,RegularAxis> RegularMesh2D;

template <>
inline Boundary<RegularMesh2D> parseBoundary<RegularMesh2D>(const std::string& boundary_desc, Manager&) { return RegularMesh2D::getBoundary(boundary_desc); }

template <>
inline Boundary<RegularMesh2D> parseBoundary<RegularMesh2D>(XMLReader& boundary_desc, Manager& env) { return RegularMesh2D::getBoundary(boundary_desc, env); }

/// Three-dimensional rectilinear mesh type
typedef RectangularMesh<3,RegularAxis> RegularMesh3D;

template <>
inline Boundary<RegularMesh3D> parseBoundary<RegularMesh3D>(const std::string& boundary_desc, Manager&) { return RegularMesh3D::getBoundary(boundary_desc); }

template <>
inline Boundary<RegularMesh3D> parseBoundary<RegularMesh3D>(XMLReader& boundary_desc, Manager& env) { return RegularMesh3D::getBoundary(boundary_desc, env); }

} // namespace plask


#endif // PLASK__REGULAR_H
