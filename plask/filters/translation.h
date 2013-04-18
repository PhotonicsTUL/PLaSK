#ifndef PLASK__FILTER__TRANSLATION_H
#define PLASK__FILTER__TRANSLATION_H

#include "base.h"
#include "../mesh/basic.h"

namespace plask {

/// Don't use this directly, use TranslatedDataSource instead.
template <typename PropertyT, PropertyType propertyType, typename SpaceType, typename VariadicTemplateTypesHolder>
struct TranslatedInnerDataSourceImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "TranslatedDataSource can't be used with single value properties (it can be use only with fields properties)");
};

/// Don't use this directly, use TranslatedDataSource instead.
template <typename PropertyT, typename SpaceType, typename... ExtraArgs>
struct TranslatedInnerDataSourceImpl< PropertyT, FIELD_PROPERTY, SpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >
: public InnerDataSource<PropertyT, SpaceType, SpaceType, SpaceType /*GeometryObjectD<SpaceType::DIM>*/, GeometryObjectD<SpaceType::DIM>>
{
    using typename InnerDataSource<PropertyT, SpaceType, SpaceType, SpaceType /*GeometryObjectD<SpaceType::DIM>*/, GeometryObjectD<SpaceType::DIM>>::Region;

    virtual boost::optional<typename PropertyT::ValueType> get(const Vec<SpaceType::DIM, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const {
        const Region* r = this->findRegion(p);
        if (r)
            return this->in(OnePointMesh<SpaceType::DIM>(p - r->inTranslation), std::forward<ExtraArgs>(extra_args)..., method)[0];
        else
            return boost::optional<typename PropertyT::ValueType>();
    }

};

/**
 * Source of data in space @p SpaceType space which read it from inner space of same type.
 */
template <typename PropertyT, typename SpaceType>
using TranslatedInnerDataSource = TranslatedInnerDataSourceImpl<PropertyT, PropertyT::propertyType, SpaceType, typename PropertyT::ExtraParams>;

}   // namespace plask

#endif // PLASK__FILTER__TRANSLATION_H
