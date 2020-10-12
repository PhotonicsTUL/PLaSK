#ifndef PLASK__FILTER__CHANGE_SPACE_SIZE_CYL_H
#define PLASK__FILTER__CHANGE_SPACE_SIZE_CYL_H

#include "base.hpp"
#include "../mesh/basic.hpp"
#include "../mesh/transformed.hpp"

namespace plask {

/// Don't use this directly, use DataFrom3DtoCyl2DSource instead.
template <typename PropertyT, PropertyType propertyType, typename VariadicTemplateTypesHolder>
struct DataFrom3DtoCyl2DSourceImpl {
    static_assert(propertyType == FIELD_PROPERTY || propertyType == MULTI_FIELD_PROPERTY,
                  "DataFrom3DtoCyl2DSource can't be used with value properties (it can be used only with fields properties)");
};

/// Don't use this directly, use DataFrom3DtoCyl2DSource instead.
template <typename PropertyT, typename... ExtraArgs>
struct DataFrom3DtoCyl2DSourceImpl<PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...>>
: public OuterDataSource<PropertyT, Geometry2DCylindrical, Geometry3D, Revolution, GeometryObjectD<3>>
{
    /// Points count for average function
    std::size_t pointsCount;

    explicit DataFrom3DtoCyl2DSourceImpl(std::size_t pointsCount = 10): pointsCount(pointsCount) {}

    /// Type of property value in output space
    typedef typename PropertyAt<PropertyT, Geometry2DCylindrical>::ValueType ValueType;

    std::function<plask::optional<ValueType>(std::size_t index)> operator()(const shared_ptr<const MeshD<2>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const override {
        const std::size_t point_count = this->pointsCount;
        auto data = this->in(
                        plask::make_shared<PointsOnCircleMeshExtend>(dst_mesh, this->inTranslation, point_count),
                        std::forward<ExtraArgs>(extra_args)..., method);
        return [point_count, data] (std::size_t index) {
            index *= point_count;
            auto sum = data[index];
            for (std::size_t i = 1; i < point_count; ++i) sum += data[index+i];
PLASK_NO_CONVERSION_WARNING_BEGIN
            return PropertyT::value3Dto2D(sum / point_count);
PLASK_NO_WARNING_END
		};
    }
};

/// Don't use this directly, use DataFrom3DtoCyl2DSource instead.
template <typename PropertyT, typename... ExtraArgs>
struct DataFrom3DtoCyl2DSourceImpl<PropertyT, MULTI_FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...>>
: public OuterDataSource<PropertyT, Geometry2DCylindrical, Geometry3D, Revolution, GeometryObjectD<3>>
{
    /// Points count for average function
    std::size_t pointsCount;

    explicit DataFrom3DtoCyl2DSourceImpl(std::size_t pointsCount = 10): pointsCount(pointsCount) {}

    /// Type of property value in output space
    typedef typename PropertyAt<PropertyT, Geometry2DCylindrical>::ValueType ValueType;

    typedef typename PropertyT::EnumType EnumType;

    std::function<plask::optional<ValueType>(std::size_t index)> operator()(EnumType n, const shared_ptr<const MeshD<2>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const override {
        const std::size_t point_count = this->pointsCount;
        auto data = this->in(n,
                        plask::make_shared<PointsOnCircleMeshExtend>(dst_mesh, this->inTranslation, point_count),
                        std::forward<ExtraArgs>(extra_args)..., method);
        return [point_count, data] (std::size_t index) {
            index *= point_count;
            auto sum = data[index];
            for (std::size_t i = 1; i < point_count; ++i) sum += data[index+i];
PLASK_NO_CONVERSION_WARNING_BEGIN
            return PropertyT::value3Dto2D(sum / point_count);
PLASK_NO_WARNING_END
        };
    }

    size_t size() const override { return this->in.size(); }
};

/**
 * Source of data in 2D space which read, and averages data from outer 3D space.
 */
template <typename PropertyT>
using DataFrom3DtoCyl2DSource = DataFrom3DtoCyl2DSourceImpl<PropertyT, PropertyT::propertyType, typename PropertyT::ExtraParams>;



/// Don't use this directly, use DataFromCyl2Dto3DSource instead.
template <typename PropertyT, PropertyType propertyType, typename VariadicTemplateTypesHolder>
struct DataFromCyl2Dto3DSourceImpl {
    static_assert(propertyType == FIELD_PROPERTY || propertyType == MULTI_FIELD_PROPERTY,
                  "DataFromCyl2Dto3DSource can't be used with value properties (it can be used only with fields properties)");
};

/// Don't use this directly, use DataFromCyl2Dto3DSource instead.
template <typename PropertyT, typename... ExtraArgs>
struct DataFromCyl2Dto3DSourceImpl<PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...>>
: public InnerDataSource<PropertyT, Geometry3D, Geometry2DCylindrical, Geometry3D /*GeometryObjectD<3>*/, Revolution>
{

    using typename InnerDataSource<PropertyT, Geometry3D, Geometry2DCylindrical, Geometry3D /*GeometryObjectD<3>*/, Revolution>::Region;

    /// Type of property value in output space
    typedef typename PropertyAt<PropertyT, Geometry3D>::ValueType ValueType;

    /// Type of property value in input space
    typedef typename PropertyAt<PropertyT, Geometry2DCylindrical>::ValueType InputValueType;

    double r_sqr_begin, r_sqr_end;

    struct LazySourceImpl {

        std::vector<LazyData<InputValueType>> dataForRegion;

        const DataFromCyl2Dto3DSourceImpl< PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >& source;

        const shared_ptr<const MeshD<3>> dst_mesh;

        /*std::tuple<ExtraArgs...> extra_args;

        InterpolationMethod method;*/

        LazySourceImpl(const DataFromCyl2Dto3DSourceImpl< PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >& source,
                       const shared_ptr<const MeshD<3>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method)
            : dataForRegion(source.regions.size()), source(source), dst_mesh(dst_mesh)/*, extra_args(extra_args...), method(method)*/
        {
            for (std::size_t region_index = 0; region_index < source.regions.size(); ++region_index)
                dataForRegion[region_index].reset(source.in(plask::make_shared<CylReductionTo2DMesh>(dst_mesh, source.regions[region_index].inTranslation), std::forward<ExtraArgs>(extra_args)..., method));
        }

        plask::optional<ValueType> operator()(std::size_t index) {
            Vec<3, double> p = dst_mesh->at(index);
            std::size_t region_index = source.findRegionIndex(p,
                        [&](const Region& r) {
                            //check if p can be in cylinder inside r
                            const Vec<3, double> v = p - r.inTranslation;  // r.inTranslation points to center of cylinder base
                            const double distance_from_center_sqr = std::fma(v.rad_p(), v.rad_p(), v.rad_r() * v.rad_r());
                            return this->source.r_sqr_begin <= distance_from_center_sqr && distance_from_center_sqr <= this->source.r_sqr_end;
                        }
            );
            if (region_index == source.regions.size())
                return plask::optional<ValueType>();

            /*if (dataForRegion[region_index].isNull())
                dataForRegion[region_index].reset(source.in(plask::make_shared<CylReductionTo2DMesh>(dst_mesh, source.regions[region_index].inTranslation), extra_args, method));*/

            return PropertyT::value2Dto3D(dataForRegion[region_index][index]);
        }

    };

    void calcConnectionParameters() override {
        InnerDataSource<PropertyT, Geometry3D, Geometry2DCylindrical, Geometry3D, Revolution>::calcConnectionParameters();
        auto child = this->inputObj->getChild();
        if (!child) {
            r_sqr_begin = r_sqr_end = 0.;
            return;
        }
        auto box = child->getBoundingBox();
        r_sqr_begin = std::max(box.lower.rad_r(), 0.0); r_sqr_begin *= r_sqr_begin;
        r_sqr_end = std::abs(box.upper.rad_r()); r_sqr_end *= r_sqr_end;
    }

    std::function<plask::optional<ValueType>(std::size_t index)> operator()(const shared_ptr<const MeshD<3>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const override {
        return LazySourceImpl(*this, dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
    }


};

/// Don't use this directly, use DataFromCyl2Dto3DSource instead.
template <typename PropertyT, typename... ExtraArgs>
struct DataFromCyl2Dto3DSourceImpl<PropertyT, MULTI_FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...>>
: public InnerDataSource<PropertyT, Geometry3D, Geometry2DCylindrical, Geometry3D /*GeometryObjectD<3>*/, Revolution>
{

    using typename InnerDataSource<PropertyT, Geometry3D, Geometry2DCylindrical, Geometry3D /*GeometryObjectD<3>*/, Revolution>::Region;

    /// Type of property value in output space
    typedef typename PropertyAt<PropertyT, Geometry3D>::ValueType ValueType;

    /// Type of property value in input space
    typedef typename PropertyAt<PropertyT, Geometry2DCylindrical>::ValueType InputValueType;

    typedef typename PropertyT::EnumType EnumType;

    double r_sqr_begin, r_sqr_end;

    struct LazySourceImpl {

        std::vector<LazyData<InputValueType>> dataForRegion;

        const DataFromCyl2Dto3DSourceImpl< PropertyT, MULTI_FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >& source;

        const shared_ptr<const MeshD<3>> dst_mesh;

        /*std::tuple<ExtraArgs...> extra_args;

        InterpolationMethod method;*/

        LazySourceImpl(const DataFromCyl2Dto3DSourceImpl< PropertyT, MULTI_FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >& source,
                       EnumType n, const shared_ptr<const MeshD<3>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method)
            : dataForRegion(source.regions.size()), source(source), dst_mesh(dst_mesh)/*, extra_args(extra_args...), method(method)*/
        {
            for (std::size_t region_index = 0; region_index < source.regions.size(); ++region_index)
                dataForRegion[region_index].reset(source.in(n, plask::make_shared<CylReductionTo2DMesh>(dst_mesh, source.regions[region_index].inTranslation), std::forward<ExtraArgs>(extra_args)..., method));
        }

        plask::optional<ValueType> operator()(std::size_t index) {
            Vec<3, double> p = dst_mesh->at(index);
            std::size_t region_index = source.findRegionIndex(p,
                        [&](const Region& r) {
                            //check if p can be in cylinder inside r
                            const Vec<3, double> v = p - r.inTranslation;  // r.inTranslation points to center of cylinder base
                            const double distance_from_center_sqr = std::fma(v.rad_p(), v.rad_p(), v.rad_r() * v.rad_r());
                            return this->source.r_sqr_begin <= distance_from_center_sqr && distance_from_center_sqr <= this->source.r_sqr_end;
                        }
            );
            if (region_index == source.regions.size())
                return plask::optional<ValueType>();

            /*if (dataForRegion[region_index].isNull())
                dataForRegion[region_index].reset(source.in(plask::make_shared<CylReductionTo2DMesh>(dst_mesh, source.regions[region_index].inTranslation), extra_args, method));*/

            return PropertyT::value2Dto3D(dataForRegion[region_index][index]);
        }

    };

    void calcConnectionParameters() override {
        InnerDataSource<PropertyT, Geometry3D, Geometry2DCylindrical, Geometry3D, Revolution>::calcConnectionParameters();
        auto child = this->inputObj->getChild();
        if (!child) {
            r_sqr_begin = r_sqr_end = 0.;
            return;
        }
        auto box = child->getBoundingBox();
        r_sqr_begin = std::max(box.lower.rad_r(), 0.0); r_sqr_begin *= r_sqr_begin;
        r_sqr_end = std::abs(box.upper.rad_r()); r_sqr_end *= r_sqr_end;
    }

    std::function<plask::optional<ValueType>(std::size_t index)> operator()(EnumType n, const shared_ptr<const MeshD<3>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const override {
        return LazySourceImpl(*this, n, dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
    }

    size_t size() const override { return this->in.size(); }
};

/**
 * Source of data in 3D space which read it from inner 2D cylindrical space (Revolution).
 */
template <typename PropertyT>
using DataFromCyl2Dto3DSource = DataFromCyl2Dto3DSourceImpl<PropertyT, PropertyT::propertyType, typename PropertyT::ExtraParams>;


}   // namespace plask

#endif // PLASK__FILTER__CHANGE_SPACE_SIZE_CYL_H
