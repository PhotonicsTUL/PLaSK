#ifndef PLASK__SPACE_H
#define PLASK__SPACE_H

#include <cmath>
using std::sin; using std::cos; using std::atan2; using std::sqrt; // TODO: use config.h here to choose proper math library

#include "vector/2d.h"
#include "vector/3d.h"

namespace plask {

/*
/// List off all available spaces
enum Spaces {
    SPACE_XY,
    SPACE_RZ,
    SPACE_XYZ
};
*/

/**
 * Structure representing 2D Cartesian space.
 *
 * Provides means of converting coordinates and vector components between local and global coordinates
**/
struct SpaceXY {

    const static int DIMS = 2;

    /// Convert local to global coordinates
    inline static Vec3<double> ABC(const Vec2<double>& xy) {
        return Vec3<double>(0., xy.x, xy.y);
    }
    /// Convert local to global coordinates
    inline static Vec3<double> ABC(double x, double y) {
        return Vec3<double>(0., x, y);
    }
    /// Convert local to global coordinates
    inline static Vec3<double> ABC(const Vec3<double>& xyz) {
        return Vec3<double>(xyz.z, xyz.x, xyz.y);
    }
    /// Convert local to global coordinates
    inline static Vec3<double> ABC(double x, double y, double z) {
        return Vec3<double>(z, x, y);
    }

    /// Convert global to local coordinates
    inline static Vec2<double> XY(const Vec3<double>& abc) {
        return Vec2<double>(abc.b, abc.c);
    }
    /// Convert global to local coordinates
    inline static Vec2<double> XY(double a, double b, double c) {
        return Vec2<double>(b, c);
    }

    /// Convert 2D vector components from local to global coordinates
    template<typename T>
    inline static const Vec3<T> inABC(const Vec2<T>& v_xy) {
        return Vec3<double>(0., v_xy.x, v_xy.y);
    }
    /// Convert 3D vector components from local to global coordinates
    template<typename T>
    inline static const Vec3<T> inABC(const Vec3<T>& v_xyz) {
        return Vec3<double>(v_xyz.z, v_xyz.x, v_xyz.y);
    }

    /// Convert vector components from global to local coordinates discarding z component
    template<typename T>
    inline static const Vec2<T> inXY(const Vec3<T>& v_abc) {
        return Vec2<double>(v_abc.b, v_abc.c);
    }

    /// Convert vector components from global to local coordinates
    template<typename T>
    inline static const Vec3<T> inXYZ(const Vec3<T>& v_abc) {
        return Vec3<double>(v_abc.b, v_abc.c, v_abc.a);
    }
};


/**
 * Structure representing 2D cylindrical space.
 *
 * Provides means of converting coordinates and vector components between local and global coordinates
**/
struct SpaceRZ {

    const static int DIMS = 2;

    /// Convert local to global coordinates
    inline static Vec3<double> ABC(const Vec2<double>& rz) {
        return Vec3<double>(rz.r, 0, rz.z);
    }
    /// Convert local to global coordinates
    inline static Vec3<double> ABC(double r, double z) {
        return Vec3<double>(r, 0., z);
    }
    /// Convert local to global coordinates
    inline static Vec3<double> ABC(const Vec3<double>& rpz) {
        return Vec3<double>(rpz.r*cos(rpz.phi), rpz.r*sin(rpz.phi), rpz.z);
    }
    /// Convert local to global coordinates
    inline static Vec3<double> ABC(double r, double phi, double z) {
        return Vec3<double>(r*cos(phi), r*sin(phi), z);
    }

    /// Convert global to local coordinates
    inline static Vec2<double> RZ(const Vec3<double>& abc) {
        if (abc.b == 0.) return Vec2<double>(abc.a, abc.c);
        else return Vec2<double>( sqrt(abc.a*abc.a + abc.b*abc.b), abc.c );
    }
    /// Convert global to local coordinates
    inline static Vec2<double> RZ(double a, double b, double c) {
        if (b == 0.) return Vec2<double>(a, c);
        else return Vec2<double>( sqrt(a*a + b*b), c );
    }

    /// Convert 2D vector components from local to global coordinates for φ = 0
    template<typename T>
    inline static const Vec3<T> inABC(const Vec2<T>& v_rz) {
        return Vec3<double>(v_rz.r, 0., v_rz.z);
    }
    /// Convert 2D vector components from local to global coordinates for any φ.
    template<typename T>
    inline static const Vec3<T> inABC(const Vec2<T>& v_rz, double phi) {
        return Vec3<double>(v_rz.r*cos(phi), v_rz.r*sin(phi), v_rz.z);
    }
    /// Convert 2D vector components from local to global coordinates in any point
    template<typename T>
    inline static const Vec3<T> inABC(const Vec2<T>& v_rz, const Vec3<double>& abc) {
        register double phi = atan2(abc.b, abc.a);
        return Vec3<double>(v_rz.r*cos(phi), v_rz.r*sin(phi), v_rz.z);
    }
    /// Convert 3D vector components from local to global coordinates for φ = 0
    template<typename T>
    inline static const Vec3<T> inABC(const Vec3<T>& v_rpz) {
        return v_rpz;
    }
    /// Convert 3D vector components from local to global coordinates for any φ.
    template<typename T>
    inline static const Vec3<T> inABC(const Vec3<T>& v_rpz, double phi) {
        register double s = sin(phi), c = cos(phi);
        return Vec3<double>( v_rpz.r * c - v_rpz.phi * s, v_rpz.r * s + v_rpz.phi * c, v_rpz.z);
    }
    /// Convert 3D vector components from local to global coordinates in any point
    template<typename T>
    inline static const Vec3<T> inABC(const Vec3<T>& v_rpz, const Vec3<double>& abc) {
        register double phi = atan2(abc.b, abc.a);
        return inABC(v_rpz, phi);
    }

    /// Convert vector components from global to local coordinates for φ = 0 discarding angular component
    template<typename T>
    inline static const Vec2<T> inRZ(const Vec3<T>& v_abc) {
        return Vec2<double>(v_abc.a, v_abc.c);
    }
    /// Convert vector components from global to local coordinates for any φ discarding angular component
    template<typename T>
    inline static const Vec2<T> inRZ(const Vec3<T>& v_abc, double phi) {
        return Vec2<double>(0., 0.); // TODO
    }
    /// Convert vector components from global to local coordinates for φ = 0
    template<typename T>
    inline static const Vec3<T> inRPZ(const Vec3<T>& v_abc) {
        return v_abc;
    }
    /// Convert vector components from global to local coordinates for any φ.
    template<typename T>
    inline static const Vec3<T> inRPZ(const Vec3<T>& v_abc, double phi) {
        return Vec3<double>(0., 0.); // TODO
    }
};


/**
 * Structure representing 3D Cartesian space.
 *
 * Provides means of converting coordinates and vector components between local and global coordinates
**/
struct SpaceXYZ {

    const static int DIMS = 3;

    /// Convert local to global coordinates
    inline static Vec3<double> ABC(const Vec3<double>& xyz) {
        return xyz;
    }
    /// Convert local to global coordinates
    inline static Vec3<double> ABC(double x, double y, double z) {
        return Vec3<double>(x, y, z);
    }

    /// Convert global to local coordinates
    inline static Vec3<double> XYZ(const Vec3<double>& abc) {
        return abc;
    }
    /// Convert global to local coordinates
    inline static Vec3<double> XYZ(double a, double b, double c) {
        return Vec3<double>(a, b, c);
    }
    
    /// Convert 3D vector components from local to global coordinates
    template<typename T>
    inline static const Vec3<T> inABC(const Vec3<T>& v_xyz) {
        return v_xyz;
    }
    /// Convert vector components from global to local coordinates
    template<typename T>
    inline static const Vec3<T> inXYZ(const Vec3<T>& v_abc) {
        return v_abc;
    }
};


/**
 * Structure representing 3D Cartesian space with coordinate system rotated to match SpaceXY
 *
 * Provides means of converting coordinates and vector components between local and global coordinates
**/
struct SpaceZXY {
    //TODO
};

} // namespace plask

#endif  //PLASK__SPACE_H
