#ifndef PLASK__FILTER__CHANGE_SPACE_SIZE_H
#define PLASK__FILTER__CHANGE_SPACE_SIZE_H

#include "base.h"

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

    CartesianMesh2DTo3D(Vec<3, double> translation, const MeshD<2>& sourceMesh, double lon)
        : translation(translation), sourceMesh(sourceMesh) {
        this->translation.lon() += lon;
    }

    CartesianMesh2DTo3D(Vec<3, double> translation, const MeshD<2>& sourceMesh)
        : translation(translation), sourceMesh(sourceMesh) {}

    virtual Vec<3, double> at(std::size_t index) const override {
        return vec3Dplus2D(translation, sourceMesh.at(index));
    }

    virtual std::size_t size() const override {
        return sourceMesh.size();
    }
};

/// Don't use this directly, use DataFrom3Dto2DSource instead.
template <typename PropertyT, PropertyType propertyType, typename VariadicTemplateTypesHolder>
struct DataFrom3Dto2DSourceImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "ChangeSpaceCartesian2Dto3DDataSource can't be used with single value properties (it can be use only with fields properties)");
};

/// Don't use this directly, use ChangeSpaceCartesian2Dto3DDataSource instead.
template <typename PropertyT, typename... ExtraArgs>
struct DataFrom3Dto2DSourceImpl< PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >
: public OuterDataSource<PropertyT, Geometry2DCartesian, Geometry3D, Extrusion, GeometryObjectD<3>>
{
    /// Points count for avarage function
    std::size_t pointsCount;

    //inLinePos in 0, inObj->getLength()
    Vec<3, double> getPointAt(const Vec<2, double>& p, double lon) {
        return vec3Dplus2D(this->inTranslation, p, lon);
    }

    //inLineRelPos in 0, 1
    Vec<3, double> getPointAtRel(const Vec<2, double>& p, double inLineRelPos) {
        return getPointAt(this->inObj->getLength() * inLineRelPos);
    }

    virtual boost::optional<typename PropertyT::ValueType> get(const Vec<2, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const override {
        if (pointsCount == 1)
            return in(getPointAtRel(0.5), std::forward<ExtraArgs>(extra_args)..., method);
        const double d = this->inObj->getLength() / pointsCount;
        return avarage(this->in(
                   PointsOnLineMesh(getPointAt(d*0.5), this->inObj->getLength()-d, pointsCount),
                   std::forward<ExtraArgs>(extra_args)...,
                   method
               ));
    }

    virtual DataVector<const typename PropertyT::ValueType> operator()(const MeshD<2>& requested_points, ExtraArgs... extra_args, InterpolationMethod method) const override {
        if (pointsCount == 1)
            return this->in(
                 CartesianMesh2DTo3D(requested_points, this->inTranslation, this->inObj->getLength() * 0.5),
                 std::forward<ExtraArgs>(extra_args)...,
                 method
             );
        DataVector<typename PropertyT::ValueType> result(requested_points.size());
        PointsOnLineMesh lineMesh;
            const double d = this->inObj->getLength() / this->pointsCount;
            lineMesh.lastPointNr = this->pointsCount - 1;
            lineMesh.longSize = this->inObj->getLength() - d;
            lineMesh.begin.lon() = this->inObj->getLength() + d * 0.5;
            for (std::size_t src_point_nr = 0; src_point_nr < result.size(); ++src_point_nr) {
                const auto v = requested_points[src_point_nr];
                lineMesh.begin.tran() = this->inTranslation.tran() + v.tran();
                lineMesh.begin.vert() = this->inTranslation.vert() + v.vert();
                result[src_point_nr] =
                        avarage(this->in(
                            lineMesh,
                            std::forward<ExtraArgs>(extra_args)...,
                            method
                        ));
            }
            return result;
    }
};

/**
 * Source of data in 2D space which read, and avarages data from outer 3D space.
 */
template <typename PropertyT>
using DataFrom3Dto2DSource = DataFrom3Dto2DSourceImpl<PropertyT, PropertyT::propertyType, typename PropertyT::ExtraParams>;



/// Don't use this directly, use DataFrom2Dto3DSourceImpl instead.
template <typename PropertyT, PropertyType propertyType, typename VariadicTemplateTypesHolder>
struct DataFrom2Dto3DSourceImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "ChangeSpaceCartesian2Dto3D can't be used with single value properties (it can be use only with fields properties)");
};

/// Don't use this directly, use TranslatedDataSource instead.
template <typename PropertyT, typename... ExtraArgs>
struct DataFrom2Dto3DSourceImpl< PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >
: public InnerDataSource<PropertyT, Geometry3D, Geometry2DCartesian, GeometryObjectD<3>, Extrusion>
{
    using typename InnerDataSource<PropertyT, Geometry3D, Geometry2DCartesian, GeometryObjectD<3>, Extrusion>::Region;

    virtual boost::optional<typename PropertyT::ValueType> get(const Vec<3, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const {
        const Region* r = this->findRegion(p);
        if (r)
            return in(vec<2>(p - r->inTranslation), std::forward<ExtraArgs>(extra_args)..., method);
        else
            return boost::optional<typename PropertyT::ValueType>();
    }

};

/**
 * Source of data in 3D space which read it from inner 2D space (Extrusion).
 */
template <typename PropertyT, typename SpaceType>
using DataFrom2Dto3DSource = DataFrom2Dto3DSourceImpl<PropertyT, PropertyT::propertyType, typename PropertyT::ExtraParams>;


}   // namespace plask

#endif // PLASK__FILTER__CHANGE_SPACE_SIZE_H
