#ifndef PLASK__FILTER__CHANGE_SPACE_SIZE_H
#define PLASK__FILTER__CHANGE_SPACE_SIZE_H

#include "base.h"
#include "../mesh/basic.h"

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

/// Don't use this directly, use DataFrom3Dto2DSource instead.
template <typename PropertyT, typename... ExtraArgs>
struct DataFrom3Dto2DSourceImpl< PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >
: public OuterDataSource<PropertyT, Geometry2DCartesian, Geometry3D, Extrusion, GeometryObjectD<3>>
{
    /// Points count for average function
    std::size_t pointsCount;

    explicit DataFrom3Dto2DSourceImpl(std::size_t pointsCount = 10): pointsCount(pointsCount) {}

    /// Type of property value in output space
    typedef typename PropertyAtSpace<PropertyT, Geometry2DCartesian>::ValueType ValueType;

    //inLinePos in 0, inputObj->getLength()
    Vec<3, double> getPointAt(const Vec<2, double>& p, double lon) const {
        return vec3Dplus2D(this->inTranslation, p, lon);
    }

    //inLineRelPos in 0, 1
    Vec<3, double> getPointAtRel(const Vec<2, double>& p, double inLineRelPos) const {
        return getPointAt(p, this->outputObj->getLength() * inLineRelPos);
    }

    virtual boost::optional<ValueType> get(const Vec<2, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const override {
        if (pointsCount == 1)
            return PropertyT::value3Dto2D(this->in(toMesh(getPointAtRel(p, 0.5)), std::forward<ExtraArgs>(extra_args)..., method)[0]);
        const double d = this->outputObj->getLength() / pointsCount;
        return PropertyT::value3Dto2D(average(this->in(
                   PointsOnLineMesh(getPointAt(p, d*0.5), this->outputObj->getLength()-d, pointsCount),
                   std::forward<ExtraArgs>(extra_args)...,
                   method
               )));
    }

    virtual DataVector<const ValueType> operator()(const MeshD<2>& requested_points, ExtraArgs... extra_args, InterpolationMethod method) const override {
        if (pointsCount == 1)
            return PropertyVec3Dto2D<PropertyT>(this->in(
                 CartesianMesh2DTo3D(this->inTranslation, requested_points, this->outputObj->getLength() * 0.5),
                 std::forward<ExtraArgs>(extra_args)...,
                 method
             ));
        DataVector<ValueType> result(requested_points.size());
        PointsOnLineMesh lineMesh;
            const double d = this->outputObj->getLength() / this->pointsCount;
            lineMesh.lastPointNr = this->pointsCount - 1;
            lineMesh.longSize = this->outputObj->getLength() - d;
            lineMesh.begin.lon() = this->outputObj->getLength() + d * 0.5;
            for (std::size_t src_point_nr = 0; src_point_nr < result.size(); ++src_point_nr) {
                const auto v = requested_points[src_point_nr];
                lineMesh.begin.tran() = this->inTranslation.tran() + v.tran();
                lineMesh.begin.vert() = this->inTranslation.vert() + v.vert();
                result[src_point_nr] =
                        PropertyT::value3Dto2D(average(this->in(
                            lineMesh,
                            std::forward<ExtraArgs>(extra_args)...,
                            method
                        )));
            }
            return result;
    }
};

/**
 * Source of data in 2D space which read, and averages data from outer 3D space.
 */
template <typename PropertyT>
using DataFrom3Dto2DSource = DataFrom3Dto2DSourceImpl<PropertyT, PropertyT::propertyType, typename PropertyT::ExtraParams>;



/// Don't use this directly, use DataFrom2Dto3DSource instead.
template <typename PropertyT, PropertyType propertyType, typename VariadicTemplateTypesHolder>
struct DataFrom2Dto3DSourceImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "DataFrom2Dto3DSource can't be used with single value properties (it can be use only with fields properties)");
};

/// Don't use this directly, use DataFrom2Dto3DSource instead.
template <typename PropertyT, typename... ExtraArgs>
struct DataFrom2Dto3DSourceImpl< PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >
: public InnerDataSource<PropertyT, Geometry3D, Geometry2DCartesian, Geometry3D /*GeometryObjectD<3>*/, Extrusion>
{
    using typename InnerDataSource<PropertyT, Geometry3D, Geometry2DCartesian, Geometry3D /*GeometryObjectD<3>*/, Extrusion>::Region;

    /// Type of property value in output space
    typedef typename PropertyAtSpace<PropertyT, Geometry3D>::ValueType ValueType;

    virtual boost::optional<ValueType> get(const Vec<3, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const {
        const Region* r = this->findRegion(p);
        if (r)
            return PropertyT::value2Dto3D(this->in(toMesh(vec<2>(p - r->inTranslation)), std::forward<ExtraArgs>(extra_args)..., method)[0]);
        else
            return boost::optional<ValueType>();
    }

};

/**
 * Source of data in 3D space which read it from inner 2D space (Extrusion).
 */
template <typename PropertyT>
using DataFrom2Dto3DSource = DataFrom2Dto3DSourceImpl<PropertyT, PropertyT::propertyType, typename PropertyT::ExtraParams>;


}   // namespace plask

#endif // PLASK__FILTER__CHANGE_SPACE_SIZE_H
