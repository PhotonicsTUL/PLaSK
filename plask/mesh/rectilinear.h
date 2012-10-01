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
typedef RectangularMesh<2, RectilinearMesh1D> RectilinearMesh2D;

template <>
inline Boundary<RectilinearMesh2D> parseBoundary<RectilinearMesh2D>(const std::string& boundary_desc, plask::BoundaryParserEnviroment) { return RectilinearMesh2D::getBoundary(boundary_desc); }

template <>
inline Boundary<RectilinearMesh2D> parseBoundary<RectilinearMesh2D>(XMLReader& boundary_desc, BoundaryParserEnviroment env) { return RectilinearMesh2D::getBoundary(boundary_desc, env); }


/// Three-dimensional rectilinear mesh type
typedef RectangularMesh<3, RectilinearMesh1D> RectilinearMesh3D;

//template <>
//inline Boundary<RectilinearMesh3D> parseBoundary<RectilinearMesh3D>(const std::string& boundary_desc) { return RectilinearMesh3D::getBoundary(boundary_desc); }

} // namespace  plask

#endif // PLASK__RECTILINEAR_H
