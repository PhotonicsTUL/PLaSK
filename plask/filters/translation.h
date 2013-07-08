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

    virtual boost::optional<ValueType> get(const Vec<SpaceType::DIM, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const {
        const Region* r = this->findRegion(p);
        if (r)
            return this->in(OnePointMesh<SpaceType::DIM>(p - r->inTranslation), std::forward<ExtraArgs>(extra_args)..., method)[0];
        else
            return boost::optional<ValueType>();
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

    virtual boost::optional<ValueType> get(const Vec<SpaceType::DIM, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const override {
        return this->in(toMesh(this->inTranslation + p), std::forward<ExtraArgs>(extra_args)..., method)[0];
    }

    virtual DataVector<const ValueType> operator()(const MeshD<SpaceType::DIM>& requested_points, ExtraArgs... extra_args, InterpolationMethod method) const override {
        return this->in(translate(requested_points, this->inTranslation), std::forward<ExtraArgs>(extra_args)..., method);
    }

};

/**
 * Source of data in space @p SpaceType space which read it from outer space of same type.
 */
template <typename PropertyT, typename SpaceType>
using TranslatedOuterDataSource = TranslatedOuterDataSourceImpl<PropertyT, PropertyT::propertyType, SpaceType, typename PropertyT::ExtraParams>;

}   // namespace plask

#endif // PLASK__FILTER__TRANSLATION_H
