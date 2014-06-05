#ifndef PLASK__FILTER__CHANGE_SPACE_SIZE_CYL_H
#define PLASK__FILTER__CHANGE_SPACE_SIZE_CYL_H

#include "base.h"
#include "../mesh/basic.h"

namespace plask {

/**
 * 3D mesh that wrap 2D mesh (sourceMesh).
 * Each point from sourceMesh is replaced by pointsCount points that lie on circle.
 * Point with index I in sourceMesh is used to creates points I * pointsCount to I * (pointsCount + 1) - 1.
 */
struct PointsOnCircleMeshExtend: public MeshD<3> {

    const shared_ptr<const MeshD<2>> sourceMesh;

    Vec<3, double> translation;

    double slice;

    std::size_t pointsCount;

    Vec<3, double> getCenterForPoint(const Vec<2, double>& p) const {
        return Vec<3, double>(this->translation.lon(), this->translation.tran(), this->translation.vert() + p.rad_z());
    }

public:

    PointsOnCircleMeshExtend(const shared_ptr<const MeshD<2>>& sourceMesh, const Vec<3, double>& translation, std::size_t pointsCount)
        : sourceMesh(sourceMesh), translation(translation), slice(PI_DOUBLED / pointsCount), pointsCount(pointsCount) {
    }

    virtual Vec<3, double> at(std::size_t index) const override {
        Vec<2, double> p = sourceMesh->at(index / pointsCount);
        const double angle = slice * (index % pointsCount);
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

    std::function<boost::optional<ValueType>(std::size_t index)> operator()(const shared_ptr<const MeshD<2>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const override {
        const std::size_t point_count = this->pointsCount;
        auto data = this->in(
                        make_shared<PointsOnCircleMeshExtend>(dst_mesh, this->inTranslation, point_count),
                        std::forward<ExtraArgs>(extra_args)..., method);
        return [=] (std::size_t index) {
            index *= point_count;
            auto sum = data[index];
            for (std::size_t i = 1; i < point_count; ++i) sum += data[index+i];
            return PropertyT::value3Dto2D(sum / point_count);
        };
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

/**
 * This class is a 2D mesh which wraps 3D mesh (@p sourceMesh), reduce each point of sourceMesh (in cylinder) to 2D and translate it by given vector (@p translation).
 */
struct CylReductionTo2DMesh: public MeshD<2> {

    Vec<3, double> translation;

    const shared_ptr<const MeshD<3>> sourceMesh;

    CylReductionTo2DMesh(const shared_ptr<const MeshD<3>> sourceMesh, const Vec<3, double>& translation)
        : translation(translation), sourceMesh(sourceMesh) {}

    virtual Vec<2, double> at(std::size_t index) const override {
        return Revolution::childVec(sourceMesh->at(index) - translation);
    }

    virtual std::size_t size() const override {
        return sourceMesh->size();
    }

};

/// Don't use this directly, use DataFromCyl2Dto3DSource instead.
template <typename PropertyT, typename... ExtraArgs>
struct DataFromCyl2Dto3DSourceImpl< PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >
: public InnerDataSource<PropertyT, Geometry3D, Geometry2DCylindrical, Geometry3D /*GeometryObjectD<3>*/, Revolution>
{

    using typename InnerDataSource<PropertyT, Geometry3D, Geometry2DCylindrical, Geometry3D /*GeometryObjectD<3>*/, Revolution>::Region;

    /// Type of property value in output space
    typedef typename PropertyAtSpace<PropertyT, Geometry3D>::ValueType ValueType;

    /// Type of property value in input space
    typedef typename PropertyAtSpace<PropertyT, Geometry2DCylindrical>::ValueType InputValueType;

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
                dataForRegion[region_index].reset(source.in(make_shared<CylReductionTo2DMesh>(dst_mesh, source.regions[region_index].inTranslation), std::forward<ExtraArgs>(extra_args)..., method));
        }

        boost::optional<ValueType> operator()(std::size_t index) {
            Vec<3, double> p = dst_mesh->at(index);
            std::size_t region_index = source.findRegionIndex(p,
                        [&](const Region& r) {
                            //check if p can be in cylinder inside r
                            const Vec<3, double> v = p - r.inTranslation;  // r.inTranslation points to center of cylinder base
                            const double radius = (r.inGeomBB.upper.lon() - r.inGeomBB.lower.lon()) * 0.5;    //TODO all regions should have same size, so this can be calc. only once
                            return std::fma(v.lon(), v.lon(), v.tran() * v.tran()) <= radius * radius;
                        }
            );
            if (region_index == source.regions.size())
                return boost::optional<ValueType>();

            /*if (dataForRegion[region_index].isNull())
                dataForRegion[region_index].reset(source.in(make_shared<CylReductionTo2DMesh>(dst_mesh, source.regions[region_index].inTranslation), extra_args, method));*/

            return PropertyT::value2Dto3D(dataForRegion[region_index][index]);
        }

    };

    std::function<boost::optional<ValueType>(std::size_t index)> operator()(const shared_ptr<const MeshD<3>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const override {
        return LazySourceImpl(*this, dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
    }


};

/**
 * Source of data in 3D space which read it from inner 2D cylindrical space (Revolution).
 */
template <typename PropertyT>
using DataFromCyl2Dto3DSource = DataFromCyl2Dto3DSourceImpl<PropertyT, PropertyT::propertyType, typename PropertyT::ExtraParams>;


}   // namespace plask

#endif // PLASK__FILTER__CHANGE_SPACE_SIZE_CYL_H
