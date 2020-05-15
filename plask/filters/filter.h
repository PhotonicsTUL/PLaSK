#ifndef PLASK__FILTER_H
#define PLASK__FILTER_H

#include "translation.h"
#include "change_space_size.h"
#include "change_space_size_cyl.h"

namespace plask {

struct FilterCommonBase: public Solver {
    template <typename... Args>
    FilterCommonBase(Args&&... args): Solver(std::forward<Args>(args)...) {}
};

/// Don't use this directly, use FilterBase or Filter instead.
template <typename PropertyT, PropertyType propertyType, typename OutputSpaceType, typename VariadicTemplateTypesHolder>
struct FilterBaseImpl {
    static_assert(propertyType == FIELD_PROPERTY || propertyType == MULTI_FIELD_PROPERTY,
                  "Filter can't be used with value properties (it can be used with field properties only)");
};

template <typename PropertyT, typename OutputSpaceType, typename... ExtraArgs>
struct FilterBaseImpl< PropertyT, FIELD_PROPERTY, OutputSpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >
    : public FilterCommonBase
{
    // one outer source (input)
    // vector of inner sources (inputs)
    // one output (provider)

    typedef typename PropertyAt<PropertyT, OutputSpaceType>::ValueType ValueType;
    typedef DataSource<PropertyT, OutputSpaceType> DataSourceT;
    typedef std::unique_ptr<DataSourceT> DataSourceTPtr;
    typedef std::function<plask::optional<ValueType>(std::size_t)> DataSourceF;

    struct FilterLazyDataImpl: public LazyDataImpl<ValueType> {

        DataSourceF outerSourceData;

        std::vector<DataSourceF> innerSourcesData;

        /*const FilterBaseImpl< PropertyT, FIELD_PROPERTY, OutputSpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >& filter;*/
        shared_ptr<const MeshD<OutputSpaceType::DIM>> dst_mesh;
        //std::tuple<ExtraArgs...> extra_args;
        //InterpolationMethod method;

        FilterLazyDataImpl(
                const FilterBaseImpl< PropertyT, FIELD_PROPERTY, OutputSpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >& filter,
                const shared_ptr<const MeshD<OutputSpaceType::DIM>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method
                )
            : innerSourcesData(filter.innerSources.size()), /*filter(filter),*/ dst_mesh(dst_mesh)/*, extra_args(extra_args...), method(method)*/ {
            for (std::size_t source_index = 0; source_index < filter.innerSources.size(); ++source_index)
                innerSourcesData[source_index] = filter.innerSources[source_index]->operator()(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
            outerSourceData = filter.outerSource->operator()(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        }

        ValueType at(std::size_t point_index) const override {
            for (std::size_t source_index = 0; source_index < innerSourcesData.size(); ++source_index) {
                //if (!innerSourcesData[source_index])
                //    innerSourcesData[source_index] = filter.innerSources[source_index]->operator()(dst_mesh, extra_args, method);
                plask::optional<ValueType> v = innerSourcesData[source_index](point_index);
                if (v) return *v;
            }
            //if (!outerSourceData) outerSourceData = filter.outerSource->operator()(dst_mesh, extra_args, method);
            return *outerSourceData(point_index);
        }

        std::size_t size() const override { return dst_mesh->size(); }

    };

protected:
    std::vector<DataSourceTPtr> innerSources;

    DataSourceTPtr outerSource;

    /// Output space in which the results are provided.
    shared_ptr<OutputSpaceType> geometry;

    template <typename SourceType>
    auto setOuterRecv(std::unique_ptr<SourceType>&& outerSource) -> decltype(outerSource->in)& {
        decltype(outerSource->in)& res = outerSource->in;
        // setOuter(std::move(outerSource)); //can't call fireChange before connecting provider to returned receiver
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

    /// Provider of filtered data.
    typename ProviderFor<PropertyT, OutputSpaceType>::Delegate out;

    /**
     * Construct solver with given output @p geometry.
     * @param geometry output geometry
     */
    FilterBaseImpl(shared_ptr<OutputSpaceType> geometry): FilterCommonBase("Filter"), geometry(geometry),
        out([&] (const shared_ptr<const MeshD<OutputSpaceType::DIM>>& dst_mesh, ExtraArgs&&... extra_args, InterpolationMethod method) -> LazyData<ValueType> {
            return this->get(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        })
    {
        setDefault(PropertyAt<PropertyT, OutputSpaceType>::getDefaultValue());
    }

    std::string getClassName() const override { return "Filter"; }

    /**
     * Get this filter output geometry.
     * \return filter geometry
     */
    shared_ptr<OutputSpaceType> getGeometry() const { return geometry; }

    /**
     * Get value provided by output of this solver.
     * @param dst_mesh
     * @param extra_args
     * @param method
     * @return
     */
    LazyData<ValueType> get(const shared_ptr<const MeshD<OutputSpaceType::DIM>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const {
        return new FilterLazyDataImpl(*this, dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        /*if (innerSources.empty())   //special case, for fast getting data from outer source
            return (*outerSource)(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        DataVector<ValueT> result(dst_mesh.size());
        NoLogging nolog;
        bool silent = outerSource && typeid(*outerSource) != typeid(ConstDataSource<PropertyT, OutputSpaceType>);
        for (std::size_t i = 0; i < result.size(); ++i) {
            // iterate over inner sources, if inner sources don't provide data, use outer source:
            plask::optional<ValueT> innerVal;
            for (const DataSourceTPtr& innerSource: innerSources) {
                innerVal = innerSource->get(dst_mesh[i], std::forward<ExtraArgs>(extra_args)..., method);
                if (innerVal) { silent = true; break; }
            }
            result[i] = *(innerVal ? innerVal : outerSource->get(dst_mesh[i], std::forward<ExtraArgs>(extra_args)..., method));
            if (silent) nolog.silence();
        }
        return result;*/
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
    void setDefault(const ValueType& value) {
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

    /**
     * Set outer or append inner input.
     * @param obj
     * @param path
     * @return
     */
    virtual ReceiverFor<PropertyT, Geometry3D>& input(GeometryObjectD<3>& obj, const PathHints* path = nullptr) = 0;

    ReceiverFor<PropertyT, Geometry3D>& input(shared_ptr<GeometryObjectD<3>> obj, const PathHints* path = nullptr) {
        return input(*obj, path);
    }

    ReceiverFor<PropertyT, Geometry3D>& input(Geometry3D& inGeom, const PathHints* path = nullptr) {
        return input(inGeom.getChild(), path);
    }

    ReceiverFor<PropertyT, Geometry3D>& input(shared_ptr<Geometry3D> inGeom, const PathHints* path = nullptr) {
        return input(inGeom->getChild(), path);
    }

    virtual ReceiverFor<PropertyT, Geometry2DCartesian>& input(Geometry2DCartesian& obj, const PathHints* path = nullptr) = 0;

    ReceiverFor<PropertyT, Geometry2DCartesian>& input(shared_ptr<Geometry2DCartesian> obj, const PathHints* path = nullptr) {
        return input(*obj, path);
    }

    virtual ReceiverFor<PropertyT, Geometry2DCylindrical>& input(Geometry2DCylindrical& obj, const PathHints* path = nullptr) = 0;

    ReceiverFor<PropertyT, Geometry2DCylindrical>& input(shared_ptr<Geometry2DCylindrical> obj, const PathHints* path = nullptr) {
        return input(*obj, path);
    }

private:

    void onSourceChange(/*Provider&, bool isDestr*/) {
        //if (!isDestr)
        out.fireChanged();
    }

    void connect(DataSourceT& in) {
        in.changed.connect(boost::bind(&FilterBaseImpl::onSourceChange, this/*, _1, _2*/));
    }

    void disconnect(DataSourceT& in) {
        in.changed.disconnect(boost::bind(&FilterBaseImpl::onSourceChange, this/*, _1, _2*/));
    }

    void disconnect(DataSourceTPtr& in) {
        if (in) disconnect(*in);
    }
};

template <typename PropertyT, typename OutputSpaceType, typename... ExtraArgs>
struct FilterBaseImpl< PropertyT, MULTI_FIELD_PROPERTY, OutputSpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >
    : public FilterCommonBase
{
    // one outer source (input)
    // vector of inner sources (inputs)
    // one output (provider)

    typedef typename PropertyAt<PropertyT, OutputSpaceType>::ValueType ValueType;
    typedef DataSource<PropertyT, OutputSpaceType> DataSourceT;
    typedef std::unique_ptr<DataSourceT> DataSourceTPtr;
    typedef std::function<plask::optional<ValueType>(std::size_t)> DataSourceF;
    typedef typename PropertyT::EnumType EnumType;

    struct FilterLazyDataImpl: public LazyDataImpl<ValueType> {

        DataSourceF outerSourceData;

        std::vector<DataSourceF> innerSourcesData;

        /*const FilterBaseImpl< PropertyT, MULTI_FIELD_PROPERTY, OutputSpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >& filter;*/
        shared_ptr<const MeshD<OutputSpaceType::DIM>> dst_mesh;
        //std::tuple<ExtraArgs...> extra_args;
        //InterpolationMethod method;

        EnumType num;
        
        FilterLazyDataImpl(
                const FilterBaseImpl< PropertyT, MULTI_FIELD_PROPERTY, OutputSpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >& filter,
                EnumType num, const shared_ptr<const MeshD<OutputSpaceType::DIM>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method
                )
            : innerSourcesData(filter.innerSources.size()), /*filter(filter),*/ dst_mesh(dst_mesh)/*, extra_args(extra_args...), method(method)*/, num(num) 
        {
            for (std::size_t source_index = 0; source_index < filter.innerSources.size(); ++source_index)
                innerSourcesData[source_index] = filter.innerSources[source_index]->operator()(num, dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
            outerSourceData = filter.outerSource->operator()(num, dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        }

        ValueType at(std::size_t point_index) const override {
            for (std::size_t source_index = 0; source_index < innerSourcesData.size(); ++source_index) {
                //if (!innerSourcesData[source_index])
                //    innerSourcesData[source_index] = filter.innerSources[source_index]->operator()(dst_mesh, extra_args, method);
                plask::optional<ValueType> v = innerSourcesData[source_index](point_index);
                if (v) return *v;
            }
            //if (!outerSourceData) outerSourceData = filter.outerSource->operator()(dst_mesh, extra_args, method);
            return *outerSourceData(point_index);
        }

        std::size_t size() const override { return dst_mesh->size(); }

    };

protected:
    std::vector<DataSourceTPtr> innerSources;

    DataSourceTPtr outerSource;

    /// Output space in which the results are provided.
    shared_ptr<OutputSpaceType> geometry;

    template <typename SourceType>
    auto setOuterRecv(std::unique_ptr<SourceType>&& outerSource) -> decltype(outerSource->in)& {
        decltype(outerSource->in)& res = outerSource->in;
        // setOuter(std::move(outerSource)); //can't call fireChange before connecting provider to returned receiver
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

    /// Provider of filtered data.
    typename ProviderFor<PropertyT, OutputSpaceType>::Delegate out;

    /**
     * Construct solver with given output @p geometry.
     * @param geometry output geometry
     */
    FilterBaseImpl(shared_ptr<OutputSpaceType> geometry): FilterCommonBase("Filter"), geometry(geometry),
        out([&] (EnumType num, const shared_ptr<const MeshD<OutputSpaceType::DIM>>& dst_mesh, ExtraArgs&&... extra_args, InterpolationMethod method) -> LazyData<ValueType> {
                return this->get(num, dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
            },
            [&] () -> size_t {
                size_t size = this->outerSource->size();
                for (const auto& inner: this->innerSources) {
                    if (inner->size() != size)
                        throw DataError("All providers in {} filter must have equal number of values", PropertyT::NAME);
                }
                return size;
            })
    {
        setDefault(PropertyAt<PropertyT, OutputSpaceType>::getDefaultValue());
    }

    std::string getClassName() const override { return "Filter"; }

    /**
     * Get this filter output geometry.
     * \return filter geometry
     */
    shared_ptr<OutputSpaceType> getGeometry() const { return geometry; }

    /**
     * Get value provided by output of this solver.
     * @param dst_mesh
     * @param extra_args
     * @param method
     * @return
     */
    LazyData<ValueType> get(EnumType num, const shared_ptr<const MeshD<OutputSpaceType::DIM>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const {
        return new FilterLazyDataImpl(*this, num, dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        /*if (innerSources.empty())   //special case, for fast getting data from outer source
            return (*outerSource)(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        DataVector<ValueT> result(dst_mesh.size());
        NoLogging nolog;
        bool silent = outerSource && typeid(*outerSource) != typeid(ConstDataSource<PropertyT, OutputSpaceType>);
        for (std::size_t i = 0; i < result.size(); ++i) {
            // iterate over inner sources, if inner sources don't provide data, use outer source:
            plask::optional<ValueT> innerVal;
            for (const DataSourceTPtr& innerSource: innerSources) {
                innerVal = innerSource->get(dst_mesh[i], std::forward<ExtraArgs>(extra_args)..., method);
                if (innerVal) { silent = true; break; }
            }
            result[i] = *(innerVal ? innerVal : outerSource->get(dst_mesh[i], std::forward<ExtraArgs>(extra_args)..., method));
            if (silent) nolog.silence();
        }
        return result;*/
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
    void setDefault(const ValueType& value) {
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

    /**
     * Set outer or append inner input.
     * @param obj
     * @param path
     * @return
     */
    virtual ReceiverFor<PropertyT, Geometry3D>& input(GeometryObjectD<3>& obj, const PathHints* path = nullptr) = 0;

    ReceiverFor<PropertyT, Geometry3D>& input(shared_ptr<GeometryObjectD<3>> obj, const PathHints* path = nullptr) {
        return input(*obj, path);
    }

    ReceiverFor<PropertyT, Geometry3D>& input(Geometry3D& inGeom, const PathHints* path = nullptr) {
        return input(inGeom.getChild(), path);
    }

    ReceiverFor<PropertyT, Geometry3D>& input(shared_ptr<Geometry3D> inGeom, const PathHints* path = nullptr) {
        return input(inGeom->getChild(), path);
    }

    virtual ReceiverFor<PropertyT, Geometry2DCartesian>& input(Geometry2DCartesian& obj, const PathHints* path = nullptr) = 0;

    ReceiverFor<PropertyT, Geometry2DCartesian>& input(shared_ptr<Geometry2DCartesian> obj, const PathHints* path = nullptr) {
        return input(*obj, path);
    }

    virtual ReceiverFor<PropertyT, Geometry2DCylindrical>& input(Geometry2DCylindrical& obj, const PathHints* path = nullptr) = 0;

    ReceiverFor<PropertyT, Geometry2DCylindrical>& input(shared_ptr<Geometry2DCylindrical> obj, const PathHints* path = nullptr) {
        return input(*obj, path);
    }

private:

    void onSourceChange(/*Provider&, bool isDestr*/) {
        //if (!isDestr)
        out.fireChanged();
    }

    void connect(DataSourceT& in) {
        in.changed.connect(boost::bind(&FilterBaseImpl::onSourceChange, this/*, _1, _2*/));
    }

    void disconnect(DataSourceT& in) {
        in.changed.disconnect(boost::bind(&FilterBaseImpl::onSourceChange, this/*, _1, _2*/));
    }

    void disconnect(DataSourceTPtr& in) {
        if (in) disconnect(*in);
    }
};

/**
 * Base class for filters which recalculate field from one space to another (OutputSpaceType).
 * @tparam PropertyT property which has type FIELD_PROPERTY
 * @tparam OutputSpaceType space of @c out provider included in this filter
 */
template <typename PropertyT, typename OutputSpaceType>
using FilterBase = FilterBaseImpl<PropertyT, PropertyT::propertyType, OutputSpaceType, typename PropertyT::ExtraParams>;

template <typename PropertyT, typename OutputSpaceType>
struct FilterImpl {};

/**
 * Filter which provides data in 3D Cartesian space.
 *
 * It can have one or more inner (2D or 3D) inputs and one outer (3D) input or default value (which is used in all points where inner inputs don't provide data).
 * @tparam PropertyT property which has type FIELD_PROPERTY
 */
template <typename PropertyT>
struct FilterImpl<PropertyT, Geometry3D>: public FilterBase<PropertyT, Geometry3D> {

    FilterImpl(shared_ptr<Geometry3D> geometry): FilterBase<PropertyT, Geometry3D>(geometry) {}

    using FilterBase<PropertyT, Geometry3D>::setOuter;

    using FilterBase<PropertyT, Geometry3D>::appendInner;

    ReceiverFor<PropertyT, Geometry3D>& input(GeometryObjectD<3>& obj, const PathHints* path = nullptr) override {
        if (obj.hasInSubtree(this->geometry->getChild(), path))
            return setOuter(obj, path);
        else
            return appendInner(obj, path);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& input(Geometry2DCartesian& innerObj, const PathHints* path = nullptr) override {
        return appendInner(innerObj, path);
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& input(Geometry2DCylindrical& innerObj, const PathHints* path = nullptr) override {
        return appendInner(innerObj, path);
    }

    ReceiverFor<PropertyT, Geometry3D>& setOuter(GeometryObjectD<3>& outerObj, const PathHints* path = nullptr) {
        std::unique_ptr< TranslatedOuterDataSource<PropertyT, Geometry3D> > source(new TranslatedOuterDataSource<PropertyT, Geometry3D>());
        source->connect(outerObj, *this->geometry->getChild(), path);
        return this->setOuterRecv(std::move(source));
    }

    ReceiverFor<PropertyT, Geometry3D>& setOuter(shared_ptr<GeometryObjectD<3>> outerObj, const PathHints* path = nullptr) {
        return setOuter(*outerObj, path);
    }

    ReceiverFor<PropertyT, Geometry3D>& setOuter(Geometry3D& outerGeom, const PathHints* path = nullptr) {
        return setOuter(outerGeom.getChild(), path);
    }

    ReceiverFor<PropertyT, Geometry3D>& setOuter(shared_ptr<Geometry3D> outerGeom, const PathHints* path = nullptr) {
        return setOuter(outerGeom->getChild(), path);
    }

    ReceiverFor<PropertyT, Geometry3D>& appendInner(GeometryObjectD<3>& innerObj, const PathHints* path = nullptr) {
        std::unique_ptr< TranslatedInnerDataSource<PropertyT, Geometry3D> > source(new TranslatedInnerDataSource<PropertyT, Geometry3D>());
        source->connect(innerObj, *this->geometry, path);
        return this->appendInnerRecv(std::move(source));
    }

    ReceiverFor<PropertyT, Geometry3D>& appendInner(shared_ptr<GeometryObjectD<3>> innerObj, const PathHints* path = nullptr) {
        return appendInner(*innerObj, path);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& appendInner2D(Extrusion& innerObj, const PathHints* path = nullptr) {
        std::unique_ptr< DataFrom2Dto3DSource<PropertyT> > source(new DataFrom2Dto3DSource<PropertyT>());
        source->connect(innerObj, *this->geometry, path);
        return this->appendInnerRecv(std::move(source));
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& appendInner2D(shared_ptr<Extrusion> innerObj, const PathHints* path = nullptr) {
        return appendInner2D(*innerObj, path);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& appendInner(Geometry2DCartesian& innerObj, const PathHints* path = nullptr) {
        return appendInner2D(innerObj.getExtrusion(), path);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& appendInner(shared_ptr<Geometry2DCartesian> innerObj, const PathHints* path = nullptr) {
        return appendInner2D(innerObj->getExtrusion(), path);
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& appendInner2D(Revolution& innerObj, const PathHints* path = nullptr) {
        std::unique_ptr< DataFromCyl2Dto3DSource<PropertyT> > source(new DataFromCyl2Dto3DSource<PropertyT>());
        source->connect(innerObj, *this->geometry, path);
        return this->appendInnerRecv(std::move(source));
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& appendInner2D(shared_ptr<Revolution> innerObj, const PathHints* path = nullptr) {
        return appendInner2D(*innerObj, path);
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& appendInner(Geometry2DCylindrical& innerObj, const PathHints* path = nullptr) {
        return appendInner2D(innerObj.getRevolution(), path);
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& appendInner(shared_ptr<Geometry2DCylindrical> innerObj, const PathHints* path = nullptr) {
        return appendInner2D(innerObj->getRevolution(), path);
    }
};

/**
 * Filter which provides data in 2D Cartesian space.
 *
 * It can have one or more inner (2D) inputs and one outer (2D or 3D) input or default value (which is used in all points where inner inputs don't provide data).
 * @tparam PropertyT property which has type FIELD_PROPERTY
 */
template <typename PropertyT>
struct FilterImpl<PropertyT, Geometry2DCartesian>: public FilterBase<PropertyT, Geometry2DCartesian> {

    FilterImpl(shared_ptr<Geometry2DCartesian> geometry): FilterBase<PropertyT, Geometry2DCartesian>(geometry) {}

    using FilterBase<PropertyT, Geometry2DCartesian>::setOuter;

    using FilterBase<PropertyT, Geometry2DCartesian>::appendInner;

    ReceiverFor<PropertyT, Geometry3D>& input(GeometryObjectD<3>& outerObj, const PathHints* path = nullptr) override {
        return setOuter(outerObj, path);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& input(Geometry2DCartesian& obj, const PathHints* path = nullptr) override {
        return input(obj.getChild(), path);
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& input(Geometry2DCylindrical&, const PathHints* = nullptr) override {
        throw Exception("Bad use of filter over Cartesian space. Cartesian geometry 2D can't contain cylindrical geometry and can't be included in cylindrical geometry.");
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& input(GeometryObjectD<2>& obj, const PathHints* path = nullptr) {
        if (obj.hasInSubtree(this->geometry->getChild(), path))
            return setOuter(obj, path);
        else
            return appendInner(obj, path);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& input(shared_ptr<GeometryObjectD<2>> obj, const PathHints* path = nullptr) {
        return input(*obj, path);
    }

    ReceiverFor<PropertyT, Geometry3D>& setOuter(GeometryObjectD<3>& outerObj, const PathHints* path = nullptr, std::size_t pointsCount = 10) {
        std::unique_ptr< DataFrom3Dto2DSource<PropertyT> > source(new DataFrom3Dto2DSource<PropertyT>(pointsCount));
        source->connect(outerObj, *this->geometry->getExtrusion(), path);
        return this->setOuterRecv(std::move(source));
    }

    ReceiverFor<PropertyT, Geometry3D>& setOuter(shared_ptr<GeometryObjectD<3>> outerObj, const PathHints* path = nullptr, std::size_t pointsCount = 10) {
        return setOuter(*outerObj, path, pointsCount);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& setOuter(GeometryObjectD<2>& outerObj, const PathHints* path = nullptr) {
        std::unique_ptr< TranslatedOuterDataSource<PropertyT, Geometry2DCartesian> > source(new TranslatedOuterDataSource<PropertyT, Geometry2DCartesian>());
        source->connect(outerObj, *this->geometry->getChild(), path);
        return this->setOuterRecv(std::move(source));
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& setOuter(shared_ptr<GeometryObjectD<2>> outerObj, const PathHints* path = nullptr) {
        return setOuter(*outerObj, path);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& setOuter(Geometry2DCartesian& outerGeom, const PathHints* path = nullptr) {
        return setOuter(outerGeom.getChild(), path);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& setOuter(shared_ptr<Geometry2DCartesian> outerGeom, const PathHints* path = nullptr) {
        return setOuter(outerGeom->getChild(), path);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& appendInner(GeometryObjectD<2>& innerObj, const PathHints* path = nullptr) {
        std::unique_ptr< TranslatedInnerDataSource<PropertyT, Geometry2DCartesian> > source(new TranslatedInnerDataSource<PropertyT, Geometry2DCartesian>());
        source->connect(innerObj, *this->geometry, path);
        return this->appendInnerRecv(std::move(source));
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& appendInner(shared_ptr<GeometryObjectD<2>> innerObj, const PathHints* path = nullptr) {
        return this->appendInner(*innerObj, path);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& appendInner(Geometry2DCartesian& innerGeom, const PathHints* path = nullptr) {
        return appendInner(innerGeom.getChild(), path);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& appendInner(shared_ptr<Geometry2DCartesian> innerGeom, const PathHints* path = nullptr) {
        return appendInner(innerGeom->getChild(), path);
    }
};

/**
 * Filter which provides data in 2D cylindrical space.
 *
 * It can have one or more inner (2D) inputs and one outer (2D or 3D) input or default value (which is used in all points where inner inputs don't provide data).
 * @tparam PropertyT property which has type FIELD_PROPERTY
 */
template <typename PropertyT>
struct FilterImpl<PropertyT, Geometry2DCylindrical>: public FilterBase<PropertyT, Geometry2DCylindrical> {

    FilterImpl(shared_ptr<Geometry2DCylindrical> geometry): FilterBase<PropertyT, Geometry2DCylindrical>(geometry) {}

    using FilterBase<PropertyT, Geometry2DCylindrical>::setOuter;

    using FilterBase<PropertyT, Geometry2DCylindrical>::appendInner;

    ReceiverFor<PropertyT, Geometry3D>& input(GeometryObjectD<3>& outerObj, const PathHints* path = nullptr) override {
        return setOuter(outerObj, path);
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& input(Geometry2DCylindrical& obj, const PathHints* path = nullptr) override {
        return input(obj.getChild(), path);
    }

    ReceiverFor<PropertyT, Geometry2DCartesian>& input(Geometry2DCartesian&, const PathHints* = nullptr) override {
        throw Exception("Bad use of filter over cylindrical space. Cylindrical geometry can't contain Cartesian geometry 2D and can't be included in Cartesian geometry 2D.");
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& input(GeometryObjectD<2>& obj, const PathHints* path = nullptr) {
        if (obj.hasInSubtree(this->geometry->getChild(), path))
            return setOuter(obj, path);
        else
            return appendInner(obj, path);
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& input(shared_ptr<GeometryObjectD<2>> obj, const PathHints* path = nullptr) {
        return input(*obj, path);
    }

    ReceiverFor<PropertyT, Geometry3D>& setOuter(GeometryObjectD<3>& outerObj, const PathHints* path = nullptr, std::size_t pointsCount = 10) {
        std::unique_ptr< DataFrom3DtoCyl2DSource<PropertyT> > source(new DataFrom3DtoCyl2DSource<PropertyT>(pointsCount));
        source->connect(outerObj, *this->geometry->getRevolution(), path);
        return this->setOuterRecv(std::move(source));
    }

    ReceiverFor<PropertyT, Geometry3D>& setOuter(shared_ptr<GeometryObjectD<3>> outerObj, const PathHints* path = nullptr, std::size_t pointsCount = 10) {
        return setOuter(*outerObj, path, pointsCount);
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& setOuter(GeometryObjectD<2>& outerObj, const PathHints* path = nullptr) {
        std::unique_ptr< TranslatedOuterDataSource<PropertyT, Geometry2DCylindrical> > source(new TranslatedOuterDataSource<PropertyT, Geometry2DCylindrical>());
        if (source->inTranslation.rad_r() != 0.0)
            throw Exception("Bad use of a filter over cylindrical space. Connection of the data sources connected with the cylindrical geometries translated in rad_r direction are not allowed.");
        source->connect(outerObj, *this->geometry->getChild(), path);
        return this->setOuterRecv(std::move(source));
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& setOuter(shared_ptr<GeometryObjectD<2>> outerObj, const PathHints* path = nullptr) {
        return setOuter(*outerObj, path);
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& setOuter(Geometry2DCylindrical& outerGeom, const PathHints* path = nullptr) {
        return setOuter(outerGeom.getChild(), path);
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& setOuter(shared_ptr<Geometry2DCylindrical> outerGeom, const PathHints* path = nullptr) {
        return setOuter(outerGeom->getChild(), path);
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& appendInner(GeometryObjectD<2>& innerObj, const PathHints* path = nullptr) {
        std::unique_ptr< TranslatedInnerDataSource<PropertyT, Geometry2DCylindrical> > source(new TranslatedInnerDataSource<PropertyT, Geometry2DCylindrical>());
        for (const auto& r: source->regions) {
            if (r.inTranslation.rad_r() != 0.0)
                throw Exception("Bad use of a filter over cylindrical space. Connection of the data sources connected with the cylindrical geometries translated in rad_r direction are not allowed.");
        }
        source->connect(innerObj, *this->geometry, path);
        return this->appendInnerRecv(std::move(source));
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& appendInner(shared_ptr<GeometryObjectD<2>> innerObj, const PathHints* path = nullptr) {
        return this->appendInner(*innerObj, path);
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& appendInner(Geometry2DCylindrical& innerGeom, const PathHints* path = nullptr) {
        return appendInner(innerGeom.getChild(), path);
    }

    ReceiverFor<PropertyT, Geometry2DCylindrical>& appendInner(shared_ptr<Geometry2DCylindrical> innerGeom, const PathHints* path = nullptr) {
        return appendInner(innerGeom->getChild(), path);
    }
};

/**
 * Filter is a special kind of Solver which "solves" the problem using another Solvers.
 *
 * It calculates its output using input of similar type and changing it in some way,
 * for example translating it from one space to another (2D -> 3D, 3D -> 2D, etc.).
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
