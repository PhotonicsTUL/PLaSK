#ifndef PLASK__FILTER__TRANSLATION_H
#define PLASK__FILTER__TRANSLATION_H

#include "base.h"
#include "../mesh/basic.h"

namespace plask {

/// Don't use this directly, use TranslatedInnerDataSource instead.
template <typename PropertyT, PropertyType propertyType, typename SpaceType, typename VariadicTemplateTypesHolder>
struct TranslatedInnerDataSourceImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "TranslatedInnerDataSource can't be used with single value properties (it can be use only with fields properties)");
};

/// Don't use this directly, use TranslatedInnerDataSource instead.
template <typename PropertyT, typename SpaceType, typename... ExtraArgs>
struct TranslatedInnerDataSourceImpl< PropertyT, FIELD_PROPERTY, SpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >
: public InnerDataSource<PropertyT, SpaceType, SpaceType, SpaceType /*GeometryObjectD<SpaceType::DIM>*/, GeometryObjectD<SpaceType::DIM>>
{
    using typename InnerDataSource<PropertyT, SpaceType, SpaceType, SpaceType /*GeometryObjectD<SpaceType::DIM>*/, GeometryObjectD<SpaceType::DIM>>::Region;

    /// Type of property value in output space
    typedef typename PropertyAtSpace<PropertyT, SpaceType>::ValueType ValueType;

    struct LazySourceImpl {

        std::vector<LazyData<ValueType>> dataForRegion;

        const TranslatedInnerDataSourceImpl< PropertyT, FIELD_PROPERTY, SpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >& source;

        const shared_ptr<const MeshD<SpaceType::DIM>> dst_mesh;

        //std::tuple<ExtraArgs...> extra_args;

        //InterpolationMethod method;

        LazySourceImpl(const TranslatedInnerDataSourceImpl< PropertyT, FIELD_PROPERTY, SpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >& source,
                       const shared_ptr<const MeshD<SpaceType::DIM>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method)
            : dataForRegion(source.regions.size()), source(source), dst_mesh(dst_mesh)/*, extra_args(extra_args...), method(method)*/
        {
            for (std::size_t region_index = 0; region_index < source.regions.size(); ++region_index)
                dataForRegion[region_index].reset(source.in(translate(dst_mesh, - source.regions[region_index].inTranslation), std::forward<ExtraArgs>(extra_args)..., method));
        }

        boost::optional<ValueType> operator()(std::size_t index) {
            std::size_t region_index = source.findRegionIndex(dst_mesh->at(index));
            if (region_index == source.regions.size())
                return boost::optional<ValueType>();

            /*if (dataForRegion[region_index].isNull())
                dataForRegion[region_index].reset(source.in(translate(dst_mesh, - source.regions[region_index].inTranslation), extra_args, method));*/

            return dataForRegion[region_index][index];
        }

    };

    std::function<boost::optional<ValueType>(std::size_t index)> operator()(const shared_ptr<const MeshD<SpaceType::DIM>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const override {
        return LazySourceImpl(*this, dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
    }

};

/**
 * Source of data in space @p SpaceType space which read it from inner space of same type.
 */
template <typename PropertyT, typename SpaceType>
using TranslatedInnerDataSource = TranslatedInnerDataSourceImpl<PropertyT, PropertyT::propertyType, SpaceType, typename PropertyT::ExtraParams>;


/// Don't use this directly, use TranslatedOuterDataSource instead.
template <typename PropertyT, PropertyType propertyType, typename SpaceType, typename VariadicTemplateTypesHolder>
struct TranslatedOuterDataSourceImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "TranslatedInnerDataSource can't be used with single value properties (it can be use only with fields properties)");
};

/// Don't use this directly, use TranslatedOuterDataSource instead.
template <typename PropertyT, typename SpaceType, typename... ExtraArgs>
struct TranslatedOuterDataSourceImpl< PropertyT, FIELD_PROPERTY, SpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >
: public OuterDataSource<PropertyT, SpaceType, SpaceType, GeometryObjectD<SpaceType::DIM>, GeometryObjectD<SpaceType::DIM>>
{
    /// Type of property value in output space
    typedef typename PropertyAtSpace<PropertyT, SpaceType>::ValueType ValueType;

    std::function<boost::optional<ValueType>(std::size_t index)> operator()(const shared_ptr<const MeshD<SpaceType::DIM>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const override {
        LazyData<ValueType> data = this->in(translate(dst_mesh, this->inTranslation), std::forward<ExtraArgs>(extra_args)..., method);
        return [=] (std::size_t index) { return data[index]; };
    }

};

/**
 * Source of data in space @p SpaceType space which read it from outer space of same type.
 */
template <typename PropertyT, typename SpaceType>
using TranslatedOuterDataSource = TranslatedOuterDataSourceImpl<PropertyT, PropertyT::propertyType, SpaceType, typename PropertyT::ExtraParams>;

}   // namespace plask

#endif // PLASK__FILTER__TRANSLATION_H
