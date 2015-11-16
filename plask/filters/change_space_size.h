#ifndef PLASK__FILTER__CHANGE_SPACE_SIZE_H
#define PLASK__FILTER__CHANGE_SPACE_SIZE_H

#include "base.h"
#include "../mesh/basic.h"

namespace plask {

/**
 * 3D mesh that wrap 2D mesh (sourceMesh).
 * It translates all points of oryginal mesh and complement lon. parameter of each point by pointsCount values.
 * Point with index I in sourceMesh is used to creates points I * pointsCount to I * (pointsCount + 1) - 1.
 */
class PLASK_API CartesianMesh2DTo3DExtend: public MeshD<3> {

    const shared_ptr<const MeshD<2>> sourceMesh;

    Vec<3, double> translation;

    double stepSize;

    /// Number of points, must be > 1
    std::size_t pointsCount;

public:

    CartesianMesh2DTo3DExtend(const shared_ptr<const MeshD<2>>& sourceMesh, const Vec<3, double>& translation, double longBegin, double lonSize, std::size_t pointsCount)
        : sourceMesh(sourceMesh), translation(translation), stepSize(lonSize / (pointsCount-1)), pointsCount(pointsCount) {
        this->translation.lon() += longBegin;
    }

    virtual Vec<3, double> at(std::size_t index) const override {
        return translation + vec(sourceMesh->at(index / pointsCount), stepSize * index);
    }

    virtual std::size_t size() const override {
        return sourceMesh->size() * pointsCount;
    }

};

/**
 * 3D mesh that wrap 2D mesh.
 * It translates all points of oryginal mesh and complement lon. parameter of each point.
 */
class PLASK_API CartesianMesh2DTo3D: public MeshD<3> {

    Vec<3, double> translation;

    const shared_ptr<const MeshD<2>> sourceMesh;

public:

    CartesianMesh2DTo3D(const shared_ptr<const MeshD<2>>& sourceMesh, Vec<3, double> translation, double lon)
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

/// Don't use this directly, use DataFrom3Dto2DSource instead.
template <typename PropertyT, PropertyType propertyType, typename VariadicTemplateTypesHolder>
struct DataFrom3Dto2DSourceImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "DataFrom3Dto2DSource can't be used with single value properties (it can be use only with fields properties)");
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

    std::function<boost::optional<ValueType>(std::size_t index)> operator()(const shared_ptr<const MeshD<2>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const override {
        if (pointsCount > 1) {
            const double total_len = this->outputObj->getLength();
            const std::size_t point_count = this->pointsCount;
            const double d = total_len / point_count;   // first step at d/2, last at total_len - d
            auto data = this->in(
                        plask::make_shared<CartesianMesh2DTo3DExtend>(dst_mesh, this->inTranslation, d * 0.5, total_len - d, point_count),
                        std::forward<ExtraArgs>(extra_args)..., method);
            return [point_count, data] (std::size_t index) {
                index *= point_count;
                auto sum = data[index];
                for (std::size_t i = 1; i < point_count; ++i) sum += data[index+i];
                return PropertyT::value3Dto2D(sum / point_count);
            };
        } else {
            auto data = this->in(
                        plask::make_shared<CartesianMesh2DTo3D>(dst_mesh, this->inTranslation, this->outputObj->getLength() * 0.5),
                        std::forward<ExtraArgs>(extra_args)..., method);
            return [data] (std::size_t index) { return PropertyT::value3Dto2D(data[index]); };
        }
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

/**
 * This class is a 2D mesh which wraps 3D mesh (@p sourceMesh), reduce each point of sourceMesh to 2D and translate it back by given vector (@p translation).
 */
//TODO better version for rectangular source (with size reduction by the size of removed axis)
struct ReductionTo2DMesh: public MeshD<2> {

    Vec<2, double> translation;

    const shared_ptr<const MeshD<3>> sourceMesh;

    ReductionTo2DMesh(const shared_ptr<const MeshD<3>> sourceMesh, const Vec<2, double>& translation)
        : translation(translation), sourceMesh(sourceMesh) {}

    ReductionTo2DMesh(const shared_ptr<const MeshD<3>> sourceMesh, const Vec<3, double>& translation)
        : translation(vec<2>(translation)), sourceMesh(sourceMesh) {}

    virtual Vec<2, double> at(std::size_t index) const override {
        return vec<2>(sourceMesh->at(index)) - translation;
    }

    virtual std::size_t size() const override {
        return sourceMesh->size();
    }

};

/// Don't use this directly, use DataFrom2Dto3DSource instead.
template <typename PropertyT, typename... ExtraArgs>
struct DataFrom2Dto3DSourceImpl< PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >
: public InnerDataSource<PropertyT, Geometry3D, Geometry2DCartesian, Geometry3D /*GeometryObjectD<3>*/, Extrusion>
{
    using typename InnerDataSource<PropertyT, Geometry3D, Geometry2DCartesian, Geometry3D /*GeometryObjectD<3>*/, Extrusion>::Region;

    /// Type of property value in output space
    typedef typename PropertyAtSpace<PropertyT, Geometry3D>::ValueType ValueType;

    /// Type of property value in input space
    typedef typename PropertyAtSpace<PropertyT, Geometry2DCartesian>::ValueType InputValueType;

    struct LazySourceImpl {

        std::vector<LazyData<InputValueType>> dataForRegion;

        const DataFrom2Dto3DSourceImpl< PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >& source;

        const shared_ptr<const MeshD<3>> dst_mesh;

        /*std::tuple<ExtraArgs...> extra_args;

        InterpolationMethod method;*/

        LazySourceImpl(const DataFrom2Dto3DSourceImpl< PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<ExtraArgs...> >& source,
                       const shared_ptr<const MeshD<3>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method)
            : dataForRegion(source.regions.size()), source(source), dst_mesh(dst_mesh)/*, extra_args(extra_args...), method(method)*/
        {
            for (std::size_t region_index = 0; region_index < source.regions.size(); ++region_index)
                dataForRegion[region_index].reset(source.in(plask::make_shared<ReductionTo2DMesh>(dst_mesh, source.regions[region_index].inTranslation), std::forward<ExtraArgs>(extra_args)..., method));
        }

        boost::optional<ValueType> operator()(std::size_t index) {
            std::size_t region_index = source.findRegionIndex(dst_mesh->at(index));
            if (region_index == source.regions.size())
                return boost::optional<ValueType>();

            /*if (dataForRegion[region_index].isNull())
                dataForRegion[region_index].reset(source.in(plask::make_shared<ReductionTo2DMesh>(dst_mesh, source.regions[region_index].inTranslation), extra_args, method));*/

            return PropertyT::value2Dto3D(dataForRegion[region_index][index]);
        }

    };

    std::function<boost::optional<ValueType>(std::size_t index)> operator()(const shared_ptr<const MeshD<3>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const override {
        return LazySourceImpl(*this, dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
    }

};

/**
 * Source of data in 3D space which read it from inner 2D space (Extrusion).
 */
template <typename PropertyT>
using DataFrom2Dto3DSource = DataFrom2Dto3DSourceImpl<PropertyT, PropertyT::propertyType, typename PropertyT::ExtraParams>;


}   // namespace plask

#endif // PLASK__FILTER__CHANGE_SPACE_SIZE_H
