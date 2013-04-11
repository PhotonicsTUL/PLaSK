#ifndef PLASK__FILTER_H
#define PLASK__FILTER_H

#include "translation.h"
#include "change_space_size.h"

namespace plask {

/// Don't use this directly, use StandardFilter instead.
template <typename PropertyT, PropertyType propertyType, typename OutputSpaceType, typename VariadicTemplateTypesHolder>
struct FilterImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "Filter can't be use with single value properties (it can be use only with fields properties)");
};

template <typename PropertyT, typename OutputSpaceType, typename... ExtraArgs>
struct FilterImpl< PropertyT, FIELD_PROPERTY, OutputSpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >
    : public Solver
{
    //one outer source (input)
    //vector if inner sources (inputs)
    //one output (provider)

    typedef typename PropertyT::ValueType ValueT;
    typedef DataSource<PropertyT, OutputSpaceType> DataSourceT;
    typedef std::unique_ptr<DataSourceT> DataSourceTPtr;

    std::vector<DataSourceTPtr> innerSources;

    DataSourceTPtr outerSource;

    shared_ptr<OutputSpaceType> outputSpace;    //shared_ptr?

    typename ProviderFor<PropertyT, OutputSpaceType>::Delegate out;

    FilterImpl(shared_ptr<OutputSpaceType> outputSpace): Solver("Filter"), outputSpace(outputSpace) {
        this->out.valueGetter = [&] (const MeshD<OutputSpaceType::DIMS>& dst_mesh, ExtraArgs&&... extra_args, InterpolationMethod method) -> DataVector<const ValueT> {
            return this->get(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        };
    }

    virtual std::string getClassName() const override { return "Filter"; }

    DataVector<const ValueT> get(const MeshD<OutputSpaceType::DIMS>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const {
        if (innerSources.empty())   //special case, for fast getting data from outer source
            return outerSource(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        DataVector<ValueT> result(dst_mesh.size());
        for (std::size_t i = 0; i < result.size(); ++i) {
            //iterate over inner sources, if inner sources don't provide data, use outer source:
            boost::optional<ValueT> innerVal;
            for (const DataSourceTPtr& innerSource: innerSources) {
                boost::optional<ValueT> innerVal = innerSource->get(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
                if (innerVal) break;
            }
            result[i] = innerVal ? *innerVal : outerSource->get(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        }
        return result;
    }



private:
    // used by source creators methods
    template <typename ReceiverT>
    ReceiverT& connect(ReceiverT& in) {
        in.providerValueChanged.connect([&] (Provider::Listener&) { out.fireChanged(); });
        return in;
    }

};



// -------- old code, to remove after implement new one ---------------------------------------

/**
 * Filter is a special kind of Solver which solves the problem using another Solver.
 *
 * It calculates its output using input of simillar type and changing it in some way,
 * for example trasnlating it from one space to another (2D -> 3D, 3D -> 2D, etc.).
 *
 * Typically filter has one input (in receiver) and one output (out provider).
 */
template <typename ReceiverType, typename ProviderType = typename ReceiverType::ProviderType::Delegate>
struct Filter1to1: public Solver {

    ReceiverType in;

    ProviderType out;

    virtual std::string getClassName() const override { return "Filter"; }

    Filter1to1(const std::string &name): Solver(name) {
        in.providerValueChanged.connect([&] (Provider::Listener&) { out.fireChanged(); });
    }
};

/// Don't use this directly, use StandardFilter instead.
template <typename PropertyT, PropertyType propertyType, typename inputSpaceType, typename outputSpaceType, typename VariadicTemplateTypesHolder>
struct StandardFilterImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "space change filter can't be use with single value properties (it can be use only with fields properties)");
};

/// Don't use this directly, use StandardFilter instead.
template <typename PropertyT, typename inputSpaceType, typename outputSpaceType, typename... _ExtraParams>
struct StandardFilterImpl<PropertyT, FIELD_PROPERTY, inputSpaceType, outputSpaceType, VariadicTemplateTypesHolder<_ExtraParams...> >
: public Filter1to1<ReceiverFor<PropertyT, inputSpaceType>, typename ProviderFor<PropertyT, outputSpaceType>::Delegate > {

    typedef typename PropertyT::ValueType ValueT;

    StandardFilterImpl(const std::string &name)
    : Filter1to1<ReceiverFor<PropertyT, inputSpaceType>, typename ProviderFor<PropertyT, outputSpaceType>::Delegate >(name) {
        this->out.valueGetter = [&] (const MeshD<outputSpaceType::DIMS>& dst_mesh, _ExtraParams&&... extra_args, InterpolationMethod method) -> DataVector<const ValueT> {
            return this->apply(dst_mesh, std::forward<_ExtraParams>(extra_args)..., method);
        };
    }

    /**
     * @param dst_mesh set of requested points
     * @param extra_args additional provider arguments
     * @return values in points describe by mesh @a dst_mesh
     */
    virtual DataVector<const ValueT> apply(const MeshD<outputSpaceType::DIMS>& dst_mesh, _ExtraParams... extra_args, InterpolationMethod method = DEFAULT_INTERPOLATION) = 0;
};



/**
 * Filter which recalculate field from one space to another.
 *
 * Subclasses should overwrite apply method which can access @c in field (of type ReceiverFor<PropertyT, inputSpaceType>) and has signature which depends from PropertyT type:
 * @code
 * virtual DataVector<const PropertyT::ValueT> apply(const MeshD<outputSpaceType::DIMS>& dst_mesh, PropertyT::ExtraParams, InterpolationMethod method) const;
 * @endcode
 *
 * @tparam PropertyT property which has type ON_MESH_PROPERTY or FIELD_PROPERTY.
 * @tparam inputSpaceType space of @c in receiver included in this filter
 * @tparam outputSpaceType space of @c out provider included in this filter
 */
template <typename PropertyT, typename inputSpaceType, typename outputSpaceType>
using StandardFilter = StandardFilterImpl<PropertyT, PropertyT::propertyType, inputSpaceType, outputSpaceType, typename PropertyT::ExtraParams>;

/*template <int DIMS, typename typeName>
using DataSource = std::function<DataVector<const typeName>(const MeshD<DIMS>& dst_mesh)>;*/





/*template <typename PropertyT>
struct ExtrusionBase: public StandardFilter<PropertyT, Geometry2DCartesian, Geometry3D> {

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
