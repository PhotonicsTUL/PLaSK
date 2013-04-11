#ifndef PLASK__FILTER__CHANGE_SPACE_SIZE_H
#define PLASK__FILTER__CHANGE_SPACE_SIZE_H

#include "filter.h"

namespace plask {

struct PointsOnLineMesh: public MeshD<3> {

    Vec<3, double> begin;

    double longSize;

    std::size_t lastPointNr;

    PointsOnLineMesh() = default;

    PointsOnLineMesh(Vec<3, double> begin, double lonSize, std::size_t pointsCount)
        : begin(begin), longSize(lonSize), lastPointNr(pointsCount-1) {}

    virtual Vec<3, double> at(std::size_t index) const override {
        Vec<3, double> ans = begin;
        ans.lon() += longSize * index / lastPointNr;
        return ans;
    }

    virtual std::size_t size() const override {
        return lastPointNr + 1;
    }

};

struct CartesianMesh2DTo3D: public MeshD<3> {

    Vec<3, double> translation;

    const MeshD<2>& sourceMesh;

    Mesh2DCartTo3D(Vec<3, double> translation, const MeshD<2>& sourceMesh, double lon)
        : translation(translation), sourceMesh(sourceMesh) {
        this->translation.lon() += lon;
    }

    Mesh2DCartTo3D(Vec<3, double> translation, const MeshD<2>& sourceMesh)
        : translation(translation), sourceMesh(sourceMesh) {}

    virtual Vec<3, double> at(std::size_t index) const override {
        return vec3Dplus2D(translation, sourceMesh.at(index));
    }

    virtual std::size_t size() const override {
        return sourceMesh.size();
    }
};

/// Don't use this directly, use ChangeSpaceCartesian2Dto3DDataSource instead.
template <typename PropertyT, PropertyType propertyType, typename VariadicTemplateTypesHolder>
struct ChangeSpaceCartesian2Dto3DDataSourceImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "ChangeSpaceCartesian2Dto3DDataSource can't be used with single value properties (it can be use only with fields properties)");
};

/// Don't use this directly, use ChangeSpaceCartesian2Dto3DDataSource instead.
template <typename PropertyT, typename... ExtraArgs>
struct ChangeSpaceCartesian2Dto3DDataSourceImpl< PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >
: public OuterDataSource<PropertyT, Geometry2DCartesian, Geometry3D, Extrusion, GeometryObjectD<3>>
{
    /// Points count for avarage function
    std::size_t pointsCount;

    //inLinePos in 0, inObj->getLength()
    Vec<3, double> getPointAt(const Vec<2, double>& p, double lon) {
        return vec3Dplus2D(inTranslation, p, lon);
    }

    //inLineRelPos in 0, 1
    Vec<3, double> getPointAtRel(const Vec<2, double>& p, double inLineRelPos) {
        return getPointAt(inObj->getLength() * inLineRelPos);
    }

    virtual boost::optional<typename PropertyT::ValueType> get(const Vec<2, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const override {
        if (pointsCount == 1)
            return in(getPointAtRel(0.5), std::forward<ExtraParams>(extra_args)..., method);
        const double d = inObj->getLength() / pointsCount;
        return avarage(this->in(
                   PointsOnLineMesh(getPointAt(d*0.5), inObj->getLength()-d, pointsCount),
                   std::forward<ExtraParams>(extra_args)...,
                   method
               ));
    }

    virtual DataVector<const typename PropertyT::ValueType> operator()(const MeshD<2>& requested_points, ExtraArgs... extra_args, InterpolationMethod method) const override {
        if (pointsCount == 1)
            return this->in(
                 CartesianMesh2DTo3D(requested_points, inTranslation, inObj->getLength() * 0.5),
                 std::forward<ExtraParams>(extra_args)...,
                 method
             );
        DataVector<typename PropertyT::ValueType> result(requested_points.size());
        PointsOnLineMesh lineMesh;
            const double d = inObj->getLength() / pointsCount;
            lineMesh.lastPointNr = pointsCount - 1;
            lineMesh.longSize = enviroment.lonSize - d;
            lineMesh.begin.lon() = enviroment.translation.lon() + d * 0.5;
            for (std::size_t src_point_nr = 0; src_point_nr < result.size(); ++src_point_nr) {
                const auto v = requested_points[src_point_nr];
                lineMesh.begin.tran() = inTranslation.tran() + v.tran();
                lineMesh.begin.vert() = inTranslation.vert() + v.vert();
                result[src_point_nr] =
                        avarage(this->in(
                            lineMesh,
                            std::forward<ExtraParams>(extra_args)...,
                            method
                        ));
            }
            return result;
        }
    }
};

template <typename PropertyT, typename SpaceType>
using ChangeSpaceCartesian2Dto3DDataSource = ChangeSpaceCartesian2Dto3DDataSourceImpl<PropertyT, PropertyT::propertyType, SpaceType, typename PropertyT::ExtraParams>;

}   // namespace plask

#endif // PLASK__FILTER__CHANGE_SPACE_SIZE_H
