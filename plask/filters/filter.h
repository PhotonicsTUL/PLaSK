#ifndef PLASK__FILTER_H
#define PLASK__FILTER_H

#include "translation.h"
#include "change_space_size.h"

namespace plask {

/// Don't use this directly, use StandardFilter instead.
template <typename PropertyT, PropertyType propertyType, typename OutputSpaceType, typename VariadicTemplateTypesHolder>
struct FilterBaseImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "Filter can't be use with single value properties (it can be use only with fields properties)");
};

template <typename PropertyT, typename OutputSpaceType, typename... ExtraArgs>
struct FilterBaseImpl< PropertyT, FIELD_PROPERTY, OutputSpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >
    : public Solver
{
    //one outer source (input)
    //vector if inner sources (inputs)
    //one output (provider)

    typedef typename PropertyT::ValueType ValueT;
    typedef DataSource<PropertyT, OutputSpaceType> DataSourceT;
    typedef std::unique_ptr<DataSourceT> DataSourceTPtr;

protected:
    std::vector<DataSourceTPtr> innerSources;

    DataSourceTPtr outerSource;

    /// Output space in which the results are privided.
    shared_ptr<OutputSpaceType> geometry;

    template <typename SourceType>
    auto setOuterRecv(std::unique_ptr<SourceType>&& outerSource) -> decltype(outerSource->in)& {
        decltype(outerSource->in)& res = outerSource->in;
        //setOuter(std::move(outerSource)); //can't call fireChange before connect provider to returned receiver
        disconnect(this->outerSource);
        this->outerSource = std::move(outerSource);
        connect(*this->outerSource);
        return res;
    }

    template <typename SourceType>
    auto appendInnerRecv(std::unique_ptr<SourceType>&& innerSource) -> decltype(innerSource->in)& {
        decltype(innerSource->in)& res = innerSource->in;
        this->innerSources.push_back(std::move(innerSource));
        connect(*this->innerSources.back());
        return res;
    }

public:

    typename ProviderFor<PropertyT, OutputSpaceType>::Delegate out;

    FilterBaseImpl(shared_ptr<OutputSpaceType> geometry): Solver("Filter"), geometry(geometry) {
        this->out.valueGetter = [&] (const MeshD<OutputSpaceType::DIM>& dst_mesh, ExtraArgs&&... extra_args, InterpolationMethod method) -> DataVector<const ValueT> {
            return this->get(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        };
    }

    virtual std::string getClassName() const override { return "Filter"; }

    DataVector<const ValueT> get(const MeshD<OutputSpaceType::DIM>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const {
        if (innerSources.empty())   //special case, for fast getting data from outer source
            return (*outerSource)(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        DataVector<ValueT> result(dst_mesh.size());
        for (std::size_t i = 0; i < result.size(); ++i) {
            //iterate over inner sources, if inner sources don't provide data, use outer source:
            boost::optional<ValueT> innerVal;
            for (const DataSourceTPtr& innerSource: innerSources) {
                innerVal = innerSource->get(dst_mesh[i], std::forward<ExtraArgs>(extra_args)..., method);
                if (innerVal) break;
            }
            result[i] = *(innerVal ? innerVal : outerSource->get(dst_mesh[i], std::forward<ExtraArgs>(extra_args)..., method));
        }
        return result;
    }

    /**
     * Set outer source to @p outerSource.
     * @param outerSource source to use in all points where inner sources don't provide values.
     */
    void setOuter(DataSourceTPtr&& outerSource) {
        disconnect(this->outerSource);
        this->outerSource = std::move(outerSource);
        connect(*this->outerSource);
        out.fireChanged();
    }

    /**
     * Set outer provider to provide constant @p value.
     * @param value value which is used in all points where inner sources don't provide values.
     */
    void setDefault(const ValueT& value) {
        disconnect(this->outerSource);
        this->outerSource.reset(new ConstDataSource<PropertyT, OutputSpaceType>(value));
        connect(*this->outerSource);
        out.fireChanged();
    }

    /**
     * Append inner data source.
     * @param innerSource inner source to add
     */
    void appendInner(DataSourceTPtr&& innerSource) {
        this->innerSources.push_back(std::move(innerSource));
        connect(*this->innerSources.back());
        out.fireChanged();
    }

private:

    void onSourceChange(Provider&, bool isDestr) {
        if (!isDestr) out.fireChanged();
    }

    void connect(DataSourceT& in) {
        in.changed.connect(boost::bind(&FilterBaseImpl::onSourceChange, this, _1, _2));
    }

    void disconnect(DataSourceT& in) {
        in.changed.disconnect(boost::bind(&FilterBaseImpl::onSourceChange, this, _1, _2));
    }

    void disconnect(DataSourceTPtr& in) {
        if (in) disconnect(*in);
    }
};

/**
 * Base class for filter which recalculate field from one space to another (OutputSpaceType).
 * @tparam PropertyT property which has type FIELD_PROPERTY
 * @tparam OutputSpaceType space of @c out provider included in this filter
 */
template <typename PropertyT, typename OutputSpaceType>
using FilterBase = FilterBaseImpl<PropertyT, PropertyT::propertyType, OutputSpaceType, typename PropertyT::ExtraParams>;

template <typename PropertyT, typename OutputSpaceType>
struct FilterImpl {};

/**
 * Filter which provides data in 3D cartesian space.
 *
 * It can have one or more inner (2D or 3D) inputs and one outer (3D) input or default value (which is used in all points where inner inputs don't provide data).
 * @tparam PropertyT property which has type FIELD_PROPERTY
 */
template <typename PropertyT>
struct FilterImpl<PropertyT, Geometry3D>: public FilterBase<PropertyT, Geometry3D> {

    FilterImpl(shared_ptr<Geometry3D> geometry): FilterBase<PropertyT, Geometry3D>(geometry) {}

    using FilterBase<PropertyT, Geometry3D>::setOuter;

    using FilterBase<PropertyT, Geometry3D>::appendInner;

    ReceiverFor<PropertyT, Geometry3D>& appendInner(GeometryObjectD<3>& innerObj, const PathHints* path = nullptr) {
        std::unique_ptr< TranslatedInnerDataSource<PropertyT, Geometry3D> > source(new TranslatedInnerDataSource<PropertyT, Geometry3D>());
        source->connect(innerObj, *this->geometry, path);
        return this->appendInnerRecv(std::move(source));
    }

    ReceiverFor<PropertyT, Geometry3D>& appendInner(shared_ptr<GeometryObjectD<3>> innerObj, const PathHints* path = nullptr) {
        return appendInner(*innerObj, path);
    }

};

/**
 * Filter which provides data in 2D cartesian space.
 *
 * It can have one or more inner (2D) inputs and one outer (2D or 3D) input or default value (which is used in all points where inner inputs don't provide data).
 * @tparam PropertyT property which has type FIELD_PROPERTY
 */
template <typename PropertyT>
struct FilterImpl<PropertyT, Geometry2DCartesian>: public FilterBase<PropertyT, Geometry2DCartesian> {

    FilterImpl(shared_ptr<Geometry2DCartesian> geometry): FilterBase<PropertyT, Geometry2DCartesian>(geometry) {}

    using FilterBase<PropertyT, Geometry2DCartesian>::setOuter;

    using FilterBase<PropertyT, Geometry2DCartesian>::appendInner;

    ReceiverFor<PropertyT, Geometry3D>& setOuter(GeometryObjectD<3>& outerObj, const PathHints* path = nullptr, std::size_t pointsCount = 10) {
        std::unique_ptr< DataFrom3Dto2DSource<PropertyT> > source(new DataFrom3Dto2DSource<PropertyT>(pointsCount));
        source->connect(outerObj, *this->geometry->getExtrusion(), path);
        return this->setOuterRecv(std::move(source));
    }

    ReceiverFor<PropertyT, Geometry3D>& setOuter(shared_ptr<GeometryObjectD<3>> outerObj, const PathHints* path = nullptr, std::size_t pointsCount = 10) {
        return setOuter(*outerObj, path, pointsCount);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& appendInner(GeometryObjectD<2>& innerObj, const PathHints* path = nullptr) {
        std::unique_ptr< TranslatedInnerDataSource<PropertyT, Geometry2DCartesian> > source(new TranslatedInnerDataSource<PropertyT, Geometry2DCartesian>());
        source->connect(innerObj, *this->geometry, path);
        return this->appendInnerRecv(std::move(source));
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& appendInner(shared_ptr<GeometryObjectD<2>> innerObj, const PathHints* path = nullptr) {
        return this->appendInner(*innerObj, path);
    }
};

// filter in 2D cylindrical space
template <typename PropertyT>
struct FilterImpl<PropertyT, Geometry2DCylindrical>: public FilterBase<PropertyT, Geometry2DCylindrical> {

    FilterImpl(shared_ptr<Geometry2DCylindrical> geometry): FilterBase<PropertyT, Geometry2DCylindrical>(geometry) {}

    using FilterBase<PropertyT, Geometry2DCylindrical>::setOuter;

    using FilterBase<PropertyT, Geometry2DCylindrical>::appendInner;
};

/**
 * Filter is a special kind of Solver which ,,solves" the problem using another Solvers,.
 *
 * It calculates its output using input of simillar type and changing it in some way,
 * for example trasnlating it from one space to another (2D -> 3D, 3D -> 2D, etc.).
 *
 * Typically filter has one or more inputs (input receiver) and one output (output provider).
 *
 * @tparam PropertyT property which has type FIELD_PROPERTY
 * @tparam OutputSpaceType space of @c out provider included in this filter
 */
template <typename PropertyT, typename OutputSpaceType>
using Filter = FilterImpl<PropertyT, OutputSpaceType>;

}   // namespace plask

#endif // PLASK__FILTER_H
