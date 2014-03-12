#ifndef PLASK__FILTER__CHANGE_SPACE_SIZE_CYL_H
#define PLASK__FILTER__CHANGE_SPACE_SIZE_CYL_H

#include "base.h"
#include "../mesh/basic.h"

namespace plask {

struct PointsOnCircleMesh: public MeshD<3> {

    Vec<3, double> center;

    double radius;

private:
    double slice;

    std::size_t pointsCount;

public:

    void setPointsCount(std::size_t pointsCount) {
        this->pointsCount = pointsCount;
        this->slice = PI_DOUBLED / pointsCount;
    }

    PointsOnCircleMesh() = default;

    PointsOnCircleMesh(Vec<3, double> center, double readius, std::size_t pointsCount)
        : center(center), radius(readius) { setPointsCount(pointsCount); }

    virtual Vec<3, double> at(std::size_t index) const override {
        const double angle = slice * index;
        return Vec<3, double>(center.lon() + radius * cos(angle), center.tran() + radius * sin(angle), center.vert());
    }

    virtual std::size_t size() const override {
        return pointsCount;
    }

};

/// Don't use this directly, use DataFrom3DtoCyl2DSource instead.
template <typename PropertyT, PropertyType propertyType, typename VariadicTemplateTypesHolder>
struct DataFrom3DtoCyl2DSourceImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "DataFrom3DtoCyl2DSource can't be used with single value properties (it can be use only with fields properties)");
};

/// Don't use this directly, use DataFrom3DtoCyl2DSource instead.
template <typename PropertyT, typename... ExtraArgs>
struct DataFrom3DtoCyl2DSourceImpl< PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >
: public OuterDataSource<PropertyT, Geometry2DCylindrical, Geometry3D, Revolution, GeometryObjectD<3>>
{
    /// Points count for average function
    std::size_t pointsCount;

    explicit DataFrom3DtoCyl2DSourceImpl(std::size_t pointsCount = 10): pointsCount(pointsCount) {}

    /// Type of property value in output space
    typedef typename PropertyAtSpace<PropertyT, Geometry2DCylindrical>::ValueType ValueType;

    Vec<3, double> getCenterForPoint(const Vec<2, double>& p) const {
        return Vec<3, double>(this->inTranslation.lon(), this->inTranslation.tran(), this->inTranslation.vert() + p.rad_z());
    }

    virtual boost::optional<ValueType> get(const Vec<2, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const override {
        return PropertyT::value3Dto2D(average(this->in(
                   PointsOnCircleMesh(getCenterForPoint(p), p.rad_r(), pointsCount),
                   std::forward<ExtraArgs>(extra_args)...,
                   method
               )));
    }

    virtual DataVector<const ValueType> operator()(const MeshD<2>& requested_points, ExtraArgs... extra_args, InterpolationMethod method) const override {
        /*if (pointsCount == 1)
            return PropertyVec3Dto2D<PropertyT>(this->in(
                 CartesianMesh2DTo3D(this->inTranslation, requested_points, this->outputObj->getLength() * 0.5),
                 std::forward<ExtraArgs>(extra_args)...,
                 method
             ));*/
        DataVector<ValueType> result(requested_points.size());
        PointsOnCircleMesh circleMesh;
        circleMesh.setPointsCount(pointsCount);
        NoLogging nolog;
        for (std::size_t src_point_nr = 0; src_point_nr < result.size(); ++src_point_nr) {
            const auto v = requested_points[src_point_nr];
            circleMesh.center = getCenterForPoint(v);
            circleMesh.radius = v.rad_r();
            result[src_point_nr] =
                    PropertyT::value3Dto2D(average(this->in(
                        circleMesh,
                        std::forward<ExtraArgs>(extra_args)...,
                        method
                    )));
            nolog.silence();
        }
        return result;
    }
};

/**
 * Source of data in 2D space which read, and averages data from outer 3D space.
 */
template <typename PropertyT>
using DataFrom3DtoCyl2DSource = DataFrom3DtoCyl2DSourceImpl<PropertyT, PropertyT::propertyType, typename PropertyT::ExtraParams>;



/// Don't use this directly, use DataFromCyl2Dto3DSource instead.
template <typename PropertyT, PropertyType propertyType, typename VariadicTemplateTypesHolder>
struct DataFromCyl2Dto3DSourceImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "DataFromCyl2Dto3DSource can't be used with single value properties (it can be use only with fields properties)");
};

/// Don't use this directly, use DataFromCyl2Dto3DSource instead.
template <typename PropertyT, typename... ExtraArgs>
struct DataFromCyl2Dto3DSourceImpl< PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >
: public InnerDataSource<PropertyT, Geometry3D, Geometry2DCylindrical, Geometry3D /*GeometryObjectD<3>*/, Revolution>
{

    using typename InnerDataSource<PropertyT, Geometry3D, Geometry2DCylindrical, Geometry3D /*GeometryObjectD<3>*/, Revolution>::Region;

    /// Type of property value in output space
    typedef typename PropertyAtSpace<PropertyT, Geometry3D>::ValueType ValueType;

    virtual boost::optional<ValueType> get(const Vec<3, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const {
        const Region* r = this->findRegion(p, [&](const Region& r) {
                //check if p can be in cylinder inside r
                const Vec<3, double> v = p - r.inTranslation;  // r.inTranslation points to center of cylinder base
                const double radius = (r.inGeomBB.upper.lon() - r.inGeomBB.lower.lon()) * 0.5;    //TODO all regions should have same size, so this can be calc. only once
                return std::fma(v.lon(), v.lon(), v.tran() * v.tran()) <= radius * radius;
        });
        if (r)
            return PropertyT::value2Dto3D(this->in(toMesh(Revolution::childVec(p - r->inTranslation)), std::forward<ExtraArgs>(extra_args)..., method)[0]);
        else
            return boost::optional<ValueType>();
    }

};

/**
 * Source of data in 3D space which read it from inner 2D cylindrical space (Revolution).
 */
template <typename PropertyT>
using DataFromCyl2Dto3DSource = DataFromCyl2Dto3DSourceImpl<PropertyT, PropertyT::propertyType, typename PropertyT::ExtraParams>;


}   // namespace plask

#endif // PLASK__FILTER__CHANGE_SPACE_SIZE_CYL_H
