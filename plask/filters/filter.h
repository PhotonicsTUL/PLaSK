#ifndef PLASK__FILTER_H
#define PLASK__FILTER_H

#include "../solver.h"
#include "../provider/providerfor.h"

namespace plask {

//FilterBlock concept: has ResultType, has ParameterType, has ExtraParameters..., and ResultType operator()(ParameterType, ExtraParameters...)
//or optional<ResultType> operator()(ParameterType, ExtraParameters...) ??
//can be Receiver
/*template <>
struct FilterBlock {
}; ??*/



/**
 * Filter is a special kind of Solver which solves the problem using another Solver.
 *
 * It calculates its output using input of simillar type and changing it in some way,
 * for example trasnlating it from one space to another (2D -> 3D, 3D -> 2D, etc.).
 *
 * Typically filter has one input (in receiver) and one output (out provider).
 */
template <typename ReceiverType, typename ProviderType = typename ReceiverType::ProviderType::Delegate>
struct Filter: public Solver {

    ReceiverType in;

    ProviderType out;

    Filter(const std::string &name): Solver(name) {
        in.providerValueChanged.connect([&] (ReceiverType&) { out.fireChange(); });
    }
};

/// Don't use this directly, use ChangeSpaceFilter instead.
template <typename PropertyT, typename ValueT, PropertyType propertyType, typename inputSpaceType, typename outputSpaceType, typename VariadicTemplateTypesHolder>
struct ChangeSpaceFilterImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "space change filter can't be use with single value properties (it can be use only with fields properties)");
};

/// Don't use this directly, use ChangeSpaceFilter instead.
template <typename PropertyT, typename ValueT, typename inputSpaceType, typename outputSpaceType, typename... _ExtraParams>
struct ChangeSpaceFilterImpl<PropertyT, ValueT, ON_MESH_PROPERTY, inputSpaceType, outputSpaceType, VariadicTemplateTypesHolder<_ExtraParams...> >
: public Filter<ReceiverFor<PropertyT, inputSpaceType>, typename ProviderFor<PropertyT, outputSpaceType>::Delegate > {

    ChangeSpaceFilterImpl(const std::string &name)
    : Filter<ReceiverFor<PropertyT, inputSpaceType>, typename ProviderFor<PropertyT, outputSpaceType>::Delegate >(name) {
        this->out.valueGetter = [&] (const MeshD<outputSpaceType::DIMS>& dst_mesh, _ExtraParams&&... extra_args) -> DataVector<const ValueT> {
            return this->apply(dst_mesh, std::forward<_ExtraParams>(extra_args)...);
        };
    }

    /**
     * @param dst_mesh set of requested points
     * @param extra_args additional provider arguments
     * @return values in points describe by mesh @a dst_mesh
     */
    virtual DataVector<const ValueT> apply(const MeshD<outputSpaceType::DIMS>& dst_mesh, _ExtraParams... extra_args) const = 0;
};

/// Don't use this directly, use ChangeSpaceFilter instead.
template <typename PropertyT, typename ValueT, typename inputSpaceType, typename outputSpaceType, typename... _ExtraParams>
struct ChangeSpaceFilterImpl<PropertyT, ValueT, FIELD_PROPERTY, inputSpaceType, outputSpaceType, VariadicTemplateTypesHolder<_ExtraParams...> >
: public Filter<ReceiverFor<PropertyT, inputSpaceType>, typename ProviderFor<PropertyT, outputSpaceType>::Delegate > {

    ChangeSpaceFilterImpl(const std::string &name)
    : Filter<ReceiverFor<PropertyT, inputSpaceType>, typename ProviderFor<PropertyT, outputSpaceType>::Delegate >(name) {
        this->out.valueGetter = [&] (const MeshD<outputSpaceType::DIMS>& dst_mesh, _ExtraParams&&... extra_args, InterpolationMethod method) -> DataVector<const ValueT> {
            return this->apply(dst_mesh, std::forward<_ExtraParams>(extra_args)..., method);
        };
    }

    /**
     * @param dst_mesh set of requested points
     * @param extra_args additional provider arguments
     * @return values in points describe by mesh @a dst_mesh
     */
    virtual DataVector<const ValueT> apply(const MeshD<outputSpaceType::DIMS>& dst_mesh, _ExtraParams... extra_args, InterpolationMethod method = DEFAULT_INTERPOLATION) const = 0;
};

/**
 * Filter which recalculate field from one space to another.
 *
 * Subclass should overwrite apply method which can access @c in field (of type ReceiverFor<PropertyT, inputSpaceType>) and has signature which depends from PropertyT type:
 * @code
 * virtual DataVector<const PropertyT::ValueT> apply(const MeshD<outputSpaceType::DIMS>& dst_mesh, PropertyT::ExtraParams[, InterpolationMethod method]) const;
 * @endcode
 * InterpolationMethod is in signature only if PropertyT type is FIELD_PROPERTY (there is no InterpolationMethod if PropertyT type is ON_MESH_PROPERTY).
 *
 * @tparam PropertyT property which has type ON_MESH_PROPERTY or FIELD_PROPERTY.
 * @tparam inputSpaceType space of @c in receiver included in this filter
 * @tparam outputSpaceType space of @c out provider included in this filter
 */
template <typename PropertyT, typename inputSpaceType, typename outputSpaceType>
using ChangeSpaceFilter = ChangeSpaceFilterImpl<PropertyT, typename PropertyT::ValueType, PropertyT::propertyType, inputSpaceType, outputSpaceType, typename PropertyT::ExtraParams>;

/**
 * inputSpaceType - must be Geometry2D...
 */
template <typename PropertyT>
struct ChangeSpaceCartesian2Dto3D: public ChangeSpaceFilter<PropertyT, Geometry3D, Geometry2DCartesian> {

    Box3D bbox; ///<calculate using geometries, should be recalculated on changes

    Vec<3, double> translation; ///<calculate using geometries, should be recalculated on changes

    //TODO geometries, path

    /// Value provided outside an extrusion.
    typename PropertyT::ValueType outsideValue;

};

/*template <typename PropertyT>
struct ExtrusionBase: public ChangeSpaceFilter<PropertyT, Geometry2DCartesian, Geometry3D> {

    typedef typename PropertyT::ValueType ValueType;

    /// Value provided outside an extrusion.
    ValueType outsideValue;

    //double from, to;  ///< in [from, to] range values are read from in

    //TODO set from, in using Extrusion place

};

template <typename PropertyT, PropertyType propertyType, typename VariadicTemplateTypesHolder>
struct ExtrusionFilterImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "extrusion filter can't be use with single value properties (it can be use only with fields properties)");
};

template <typename PropertyT, typename... _ExtraParams>
struct ExtrusionFilterImpl<PropertyT, ON_MESH_PROPERTY, VariadicTemplateTypesHolder<_ExtraParams...> >: public ExtrusionBase<PropertyT> {

    virtual DataVector<const typename PropertyT::ValueType> apply(const MeshD<3>& dst_mesh, _ExtraParams... extra_args) const {
        //TODO
    }

};

template <typename PropertyT, typename... _ExtraParams>
struct ExtrusionFilterImpl<PropertyT, FIELD_PROPERTY, VariadicTemplateTypesHolder<_ExtraParams...> >: public ExtrusionBase<PropertyT> {

    virtual DataVector<const typename PropertyT::ValueType> apply(const MeshD<3>& dst_mesh, _ExtraParams... extra_args, InterpolationMethod method = DEFAULT_INTERPOLATION) const {
        //TODO
    }

};

template <typename PropertyT, typename inputSpaceType, typename outputSpaceType>
using ExtrusionFilter = ExtrusionFilterImpl<PropertyT, PropertyT::propertyType, typename PropertyT::ExtraParams>;*/

//TODO 3D -> 2D reduction by using constant extra coordinate
//przekrój


}   // namespace plask

#endif // PLASK__FILTER_H
