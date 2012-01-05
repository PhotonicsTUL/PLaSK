#ifndef PLASK__SPACE_H
#define PLASK__SPACE_H

#include <cmath>
// TODO: use configured math library

#include "vec.h"

namespace plask {

/**
 * Structure representing 2D Cartesian space.
 *
 * Provides means of converting coordinates and vector components between local and global coordinates
 */
struct SpaceXY {

    enum  { DIMS = 2 };

    typedef Vec<2,double> CoordsType;

    // /// Convert local to global coordinates
    // inline static Vec<3,double> ABC(const Vec<2,double>& xy, double z=0.) {
    //     return Vec<3,double>(z, xy.x, xy.y);
    // }
    // /// Convert local to global coordinates
    // inline static Vec<3,double> ABC(const Vec<3,double>& xyz) {
    //     return Vec<3,double>(xyz.z, xyz.x, xyz.y);
    // }
    // /// Convert local to global coordinates
    // inline static Vec<3,double> ABC(double x, double y, double z=0.) {
    //     return Vec<3,double>(z, x, y);
    // }
    //
    // /// Convert global to local coordinates
    // inline static Vec<2,double> XY(const Vec<3,double>& abc) {
    //     return Vec<2,double>(abc.b, abc.c);
    // }
    // /// Convert global to local coordinates
    // inline static Vec<2,double> XY(double a, double b, double c) {
    //     return Vec<2,double>(b, c);
    // }
    // /// Convert global to local coordinates (unified name)
    // template<typename ...Args>
    // inline static Vec<2,double> local(Args&&... args) {
    //     return XY(args...);
    // }
    //
    // /// Convert 2D vector components from local to global coordinates
    // template<typename T>
    // inline static const Vec<3,T> inABC(const Vec<2,T>& v_xy) {
    //     return Vec<3,double>(0., v_xy.x, v_xy.y);
    // }
    // /// Convert 3D vector components from local to global coordinates
    // template<typename T>
    // inline static const Vec<3,T> inABC(const Vec<3,T>& v_xyz) {
    //     return Vec<3,double>(v_xyz.z, v_xyz.x, v_xyz.y);
    // }
    //
    // /// Convert vector components from global to local coordinates discarding z component
    // template<typename T>
    // inline static const Vec<2,T> inXY(const Vec<3,T>& v_abc) {
    //     return Vec<2,double>(v_abc.b, v_abc.c);
    // }
    //
    // /// Convert vector components from global to local coordinates
    // template<typename T>
    // inline static const Vec<3,T> inXYZ(const Vec<3,T>& v_abc) {
    //     return Vec<3,double>(v_abc.b, v_abc.c, v_abc.a);
    // }
};


/**
 * Structure representing 2D cylindrical space.
 *
 * Provides means of converting coordinates and vector components between local and global coordinates
 */
struct SpaceRZ {

    enum { DIMS = 2 };

    typedef Vec<2,double> CoordsType;

    // /// Convert local to global coordinates
    // inline static Vec<3,double> ABC(const Vec<2,double>& rz) {
    //     return Vec<3,double>(rz.r, 0, rz.z);
    // }
    // /// Convert local to global coordinates
    // inline static Vec<3,double> ABC(double r, double z) {
    //     return Vec<3,double>(r, 0., z);
    // }
    // /// Convert local to global coordinates
    // inline static Vec<3,double> ABC(const Vec<3,double>& abc) {
    //     return Vec<3,double>(abc.r*cos(abc.phi), abc.r*sin(abc.phi), abc.z);
    // }
    // /// Convert local to global coordinates
    // inline static Vec<3,double> ABC(double r, double phi, double z) {
    //     return Vec<3,double>(r*cos(phi), r*sin(phi), z);
    // }
    // /// Convert local to global coordinates
    // inline static Vec<3,double> ABC(const Vec<2,double>& rz, double phi) {
    //     return Vec<3,double>(rz.r*cos(phi), rz.r*sin(phi), rz.z);
    // }
    //
    // /// Convert global to local coordinates
    // inline static Vec<2,double> RZ(const Vec<3,double>& abc) {
    //     if (abc.b == 0.) return Vec<2,double>(abc.a, abc.c);
    //     else return Vec<2,double>( sqrt(abc.a*abc.a + abc.b*abc.b), abc.c );
    // }
    // /// Convert global to local coordinates
    // inline static Vec<2,double> RZ(double a, double b, double c) {
    //     if (b == 0.) return Vec<2,double>(a, c);
    //     else return Vec<2,double>( sqrt(a*a + b*b), c );
    // }
    // /// Convert global to local coordinates (unified name)
    // template<typename ...Args>
    // inline static Vec<2,double> local(Args&&... args) {
    //     return RZ(args...);
    // }
    //
    // /// Convert 2D vector components from local to global coordinates for φ = 0
    // template<typename T>
    // inline static const Vec<3,T> inABC(const Vec<2,T>& v_rz) {
    //     return Vec<3,double>(v_rz.r, 0., v_rz.z);
    // }
    // /// Convert 2D vector components from local to global coordinates for any φ.
    // template<typename T>
    // inline static const Vec<3,T> inABC(const Vec<2,T>& v_rz, double phi) {
    //     return Vec<3,double>(v_rz.r*cos(phi), v_rz.r*sin(phi), v_rz.z);
    // }
    // /// Convert 2D vector components from local to global coordinates in any point
    // template<typename T>
    // inline static const Vec<3,T> inABC(const Vec<2,T>& v_rz, const Vec<3,double>& abc) {
    //     register double phi = atan2(abc.b, abc.a);
    //     return Vec<3,double>(v_rz.r*cos(phi), v_rz.r*sin(phi), v_rz.z);
    // }
    // /// Convert 3D vector components from local to global coordinates for φ = 0
    // template<typename T>
    // inline static const Vec<3,T> inABC(const Vec<3,T>& v_abc) {
    //     return v_abc;
    // }
    // /// Convert 3D vector components from local to global coordinates for any φ.
    // template<typename T>
    // inline static const Vec<3,T> inABC(const Vec<3,T>& v_abc, double phi) {
    //     register double s = sin(phi), c = cos(phi);
    //     return Vec<3,double>( v_abc.r*c - v_abc.phi*s, v_abc.r*s + v_abc.phi*c, v_abc.z);
    // }
    // /// Convert 3D vector components from local to global coordinates in any point
    // template<typename T>
    // inline static const Vec<3,T> inABC(const Vec<3,T>& v_abc, const Vec<3,double>& abc) {
    //     register double phi = atan2(abc.b, abc.a);
    //     return inABC(v_abc, phi);
    // }
    //
    // /// Convert vector components from global to local coordinates for φ = 0 discarding angular component
    // template<typename T>
    // inline static const Vec<2,T> inRZ(const Vec<3,T>& v_abc) {
    //     return Vec<2,double>(v_abc.a, v_abc.c);
    // }
    // /// Convert vector components from global to local coordinates for any φ discarding angular component
    // template<typename T>
    // inline static const Vec<2,T> inRZ(const Vec<3,T>& v_abc, double phi) {
    //     register double s = sin(phi), c = cos(phi);
    //     return Vec<2,double>( v_abc.a*c + v_abc.b*s, v_abc.c );
    // }
    // /// Convert vector components from global to local coordinates for φ = 0
    // template<typename T>
    // inline static const Vec<3,T> inRPZ(const Vec<3,T>& v_abc) {
    //     return v_abc;
    // }
    // /// Convert vector components from global to local coordinates for any φ.
    // template<typename T>
    // inline static const Vec<3,T> inRPZ(const Vec<3,T>& v_abc, double phi) {
    //     register double s = sin(phi), c = cos(phi);
    //     return Vec<3,double>( v_abc.a*c + v_abc.b*s, -v_abc.a*s + v_abc.b*c, v_abc.c );
    // }
};


/**
 * Structure representing 3D Cartesian space.
 *
 * Provides means of converting coordinates and vector components between local and global coordinates
 */
struct SpaceXYZ {

    enum { DIMS = 3 };

    typedef Vec<2,double> CoordsType;

    // /// Convert local to global coordinates
    // inline static Vec<3,double> ABC(const Vec<3,double>& xyz) {
    //     return xyz;
    // }
    // /// Convert local to global coordinates
    // inline static Vec<3,double> ABC(double x, double y, double z) {
    //     return Vec<3,double>(x, y, z);
    // }
    //
    // /// Convert global to local coordinates
    // inline static Vec<3,double> XYZ(const Vec<3,double>& abc) {
    //     return abc;
    // }
    // /// Convert global to local coordinates
    // inline static Vec<3,double> XYZ(double a, double b, double c) {
    //     return Vec<3,double>(a, b, c);
    // }
    // /// Convert global to local coordinates (unified name)
    // template<typename ...Args>
    // inline static Vec<2,double> local(Args&&... args) {
    //     return XYZ(args...);
    // }
    //
    // /// Convert 3D vector components from local to global coordinates
    // template<typename T>
    // inline static const Vec<3,T> inABC(const Vec<3,T>& v_xyz) {
    //     return v_xyz;
    // }
    // /// Convert vector components from global to local coordinates
    // template<typename T>
    // inline static const Vec<3,T> inXYZ(const Vec<3,T>& v_abc) {
    //     return v_abc;
    // }
};

} // namespace plask

#endif  //PLASK__SPACE_H
