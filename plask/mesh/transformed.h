#ifndef PLASK__MESH_TRANSFORMED_H
#define PLASK__MESH_TRANSFORMED_H

#include "mesh.h"
#include "../geometry/space.h"

namespace plask {

/**
 * This class is a 2D mesh which wraps 3D mesh (@p sourceMesh), reduce each point of sourceMesh to 2D and translate it back by given vector (@p translation).
 */
//TODO better version for rectangular source (with size reduction by the size of removed axis)
struct ReductionTo2DMesh: public MeshD<2> {

    /// Source geometry
    typedef Geometry3D SourceGeometry;

    /// Target geometry
    typedef Geometry2DCartesian TargetGeometry;

    /// Number of source dimensions
    enum { SRC_DIM = 3 };

    Vec<2,double> translation;

    const shared_ptr<const MeshD<3>> sourceMesh;

    ReductionTo2DMesh(const shared_ptr<const MeshD<3>> sourceMesh, const Vec<2,double>& translation=Primitive<2>::ZERO_VEC)
        : translation(translation), sourceMesh(sourceMesh) {}

    ReductionTo2DMesh(const shared_ptr<const MeshD<3>> sourceMesh, const Vec<3,double>& translation)
        : translation(vec<2>(translation)), sourceMesh(sourceMesh) {}

    virtual Vec<2, double> at(std::size_t index) const override {
        return vec<2>(sourceMesh->at(index)) - translation;
    }

    virtual std::size_t size() const override {
        return sourceMesh->size();
    }
};


/**
 * This class is a 2D mesh which wraps 3D mesh (@p sourceMesh), reduce each point of sourceMesh (in cylinder) to 2D and translate it by given vector (@p translation).
 */
struct PLASK_API CylReductionTo2DMesh: public MeshD<2> {

    /// Source geometry
    typedef Geometry3D SourceGeometry;

    /// Target geometry
    typedef Geometry2DCylindrical TargetGeometry;

    /// Number of source dimensions
    enum { SRC_DIM = 3 };

    Vec<3,double> translation;

    const shared_ptr<const MeshD<3>> sourceMesh;

    CylReductionTo2DMesh(const shared_ptr<const MeshD<3>> sourceMesh, const Vec<3,double>& translation=Primitive<3>::ZERO_VEC)
        : translation(translation), sourceMesh(sourceMesh) {}

    virtual Vec<2, double> at(std::size_t index) const override {
        return Revolution::childVec(sourceMesh->at(index) - translation);
    }

    virtual std::size_t size() const override {
        return sourceMesh->size();
    }
};


/**
 * 3D mesh that wrap 2D mesh.
 * It translates all points of original mesh and complement lon. parameter of each point.
 */
class PLASK_API CartesianMesh2DTo3D: public MeshD<3> {

    Vec<3,double> translation;

    const shared_ptr<const MeshD<2>> sourceMesh;

public:

    /// Source geometry
    typedef Geometry2DCartesian SourceGeometry;

    /// Target geometry
    typedef Geometry3D TargetGeometry;

    /// Number of source dimensions
    enum { SRC_DIM = 2 };

    CartesianMesh2DTo3D(const shared_ptr<const MeshD<2>>& sourceMesh, Vec<3,double> translation=Primitive<3>::ZERO_VEC, double lon=0)
        : translation(translation), sourceMesh(sourceMesh) {
        this->translation.lon() += lon;
    }

    virtual Vec<3, double> at(std::size_t index) const override {
        return vec3Dplus2D(translation, sourceMesh->at(index));
    }

    virtual std::size_t size() const override {
        return sourceMesh->size();
    }
};


/**
 * 3D mesh that wrap 2D mesh (sourceMesh).
 * It translates all points of original mesh and complement lon. parameter of each point by pointsCount values.
 * Point with index I in sourceMesh is used to creates points I * pointsCount to I * (pointsCount + 1) - 1.
 */
class PLASK_API CartesianMesh2DTo3DExtend: public MeshD<3> {

    const shared_ptr<const MeshD<2>> sourceMesh;

    Vec<3,double> translation;

    double stepSize;

    /// Number of points, must be > 1
    std::size_t pointsCount;

public:

    /// Source geometry
    typedef Geometry2DCartesian SourceGeometry;

    /// Target geometry
    typedef Geometry3D TargetGeometry;

    /// Number of source dimensions
    enum { SRC_DIM = 2 };

    CartesianMesh2DTo3DExtend(const shared_ptr<const MeshD<2>>& sourceMesh, const Vec<3,double>& translation, double longBegin, double lonSize, std::size_t pointsCount=10)
        : sourceMesh(sourceMesh), translation(translation), stepSize(lonSize / double(pointsCount-1)), pointsCount(pointsCount) {
        this->translation.lon() += longBegin;
    }

    virtual Vec<3, double> at(std::size_t index) const override {
        return translation + vec(sourceMesh->at(index / pointsCount), stepSize * double(index));
    }

    virtual std::size_t size() const override {
        return sourceMesh->size() * pointsCount;
    }
};


/**
 * 3D mesh that wrap 2D mesh (sourceMesh).
 * Each point from sourceMesh is replaced by pointsCount points that lie on circle.
 * Point with index I in sourceMesh is used to creates points I * pointsCount to I * (pointsCount + 1) - 1.
 */
struct PLASK_API PointsOnCircleMeshExtend: public MeshD<3> {

    const shared_ptr<const MeshD<2>> sourceMesh;

    Vec<3,double> translation;

    double slice;

    std::size_t pointsCount;

    Vec<3, double> getCenterForPoint(const Vec<2, double>& p) const {
        return Vec<3, double>(this->translation.lon(), this->translation.tran(), this->translation.vert() + p.rad_z());
    }

public:

    /// Source geometry
    typedef Geometry2DCylindrical SourceGeometry;

    /// Target geometry
    typedef Geometry3D TargetGeometry;

    /// Number of source dimensions
    enum { SRC_DIM = 2 };

    PointsOnCircleMeshExtend(const shared_ptr<const MeshD<2>>& sourceMesh, const Vec<3, double>& translation=Primitive<3>::ZERO_VEC, std::size_t pointsCount=18)
        : sourceMesh(sourceMesh), translation(translation), slice(PI_DOUBLED / double(pointsCount)), pointsCount(pointsCount) {
    }

    virtual Vec<3, double> at(std::size_t index) const override {
        Vec<2, double> p = sourceMesh->at(index / pointsCount);
        const double angle = slice * double(index % pointsCount);
        return Vec<3, double>(
                    this->translation.lon()  +  p.rad_r() * cos(angle),
                    this->translation.tran() +  p.rad_r() * sin(angle),
                    this->translation.vert() +  p.rad_z()
        );
    }

    virtual std::size_t size() const override {
        return sourceMesh->size() * pointsCount;
    }
};

} // namespace plask

#endif // PLASK__MESH_TRANSFORMED_H