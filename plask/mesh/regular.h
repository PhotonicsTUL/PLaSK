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

template <>
inline Boundary<RegularMesh2D> parseBoundary<RegularMesh2D>(const std::string& boundary_desc, BoundaryParserEnviroment) { return RegularMesh2D::getBoundary(boundary_desc); }

template <>
inline Boundary<RegularMesh2D> parseBoundary<RegularMesh2D>(XMLReader& boundary_desc, BoundaryParserEnviroment env) { return RegularMesh2D::getBoundary(boundary_desc, env); }

/// Three-dimensional rectilinear mesh type
typedef RectangularMesh<3,RegularMesh1D> RegularMesh3D;

template <>
inline Boundary<RegularMesh3D> parseBoundary<RegularMesh3D>(const std::string& boundary_desc, BoundaryParserEnviroment) { return RegularMesh3D::getBoundary(boundary_desc); }

template <>
inline Boundary<RegularMesh3D> parseBoundary<RegularMesh3D>(XMLReader& boundary_desc, BoundaryParserEnviroment env) { return RegularMesh3D::getBoundary(boundary_desc, env); }

} // namespace plask


#endif // PLASK__REGULAR_H
