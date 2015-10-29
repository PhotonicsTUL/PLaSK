#ifndef PLASK__FILTER__BASE_H
#define PLASK__FILTER__BASE_H

#include <vector>
#include <memory>

#include "../solver.h"
#include "../provider/providerfor.h"

namespace plask {

/// Don't use this directly, use DataSource instead.
template <typename PropertyT, PropertyType propertyType, typename OutputSpaceType, typename VariadicTemplateTypesHolder>
struct DataSourceImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "space change filter data sources can't be use with single value properties (it can be use only with fields properties)");
};

//This class is simillar to field provider, but in each point it returns optional value
template <typename PropertyT, typename OutputSpaceT, typename... ExtraArgs>
class DataSourceImpl<PropertyT, FIELD_PROPERTY, OutputSpaceT, VariadicTemplateTypesHolder<ExtraArgs...>>
//: public FieldProvider<boost::optional<typename PropertyAtSpace<PropertyT, OutputSpaceType>::ValueType>, OutputSpaceType, ExtraArgs...>    //inharistance only for change signal, not neccessery
{

    //shared_ptr<OutputSpaceType> destinationSpace;   //should be stored here? maybe only connection...

public:

    typedef OutputSpaceT OutputSpaceType;

    /**
     * Signal called when source has been changed.
     */
    boost::signals2::signal<void()> changed;

    /*shared_ptr<OutputSpaceType> getDestinationSpace() const { return destinationSpace; }

    virtual void setDestinationSpace(shared_ptr<OutputSpaceType>) { this->destinationSpace = destinationSpace; }*/

    /*
     * Check if this source can provide value for given point.
     * @param p point, in outer space coordinates
     * @return @c true only if this can provide data in given point @p p
     */
    //virtual bool canProvide(const Vec<OutputSpaceType::DIM, double>& p) const = 0;

    /// Type of property value in output space
    typedef typename PropertyAtSpace<PropertyT, OutputSpaceType>::ValueType ValueType;

    /*
     * Check if this source can provide value for given point and eventualy return this value.
     * @param p point (in outer space coordinates)
     * @param extra_args
     * @param method interpolation method to use
     * @return value in point @p, set only if this can provide data in given point @p p
     */
   // virtual boost::optional<ValueType> get(const Vec<OutputSpaceType::DIM, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const = 0;

   // virtual ValueT get(const Vec<OutputSpaceType::DIM, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const = 0;

    //virtual LazyData<boost::optional<ValueType>> operator()(const MeshD<OutputSpaceType::DIM>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const = 0;

    virtual std::function<boost::optional<ValueType>(std::size_t index)> operator()(const shared_ptr<const MeshD<OutputSpaceType::DIM>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const = 0;

    inline std::function<boost::optional<ValueType>(std::size_t index)> operator()(const shared_ptr<const MeshD<OutputSpaceType::DIM>>& dst_mesh, std::tuple<ExtraArgs...> extra_args, InterpolationMethod method) const {
        typedef std::tuple<ExtraArgs...> Tuple;
        return apply_tuple(dst_mesh, method, std::forward<Tuple>(extra_args), make_seq_indices<0, sizeof...(ExtraArgs)>{});
    }

private:
    template <typename T,  template <std::size_t...> class I, std::size_t... Indices>
    inline std::function<boost::optional<ValueType>(std::size_t index)> apply_tuple(const shared_ptr<const MeshD<OutputSpaceType::DIM>>& dst_mesh, InterpolationMethod method, T&& t, I<Indices...>) const {
      return this->operator()(dst_mesh, std::get<Indices>(std::forward<T>(t))..., method);
    }

};

//This class is simillar to field provider, but in each point it returns optional value
template <typename PropertyT, typename OutputSpaceT, typename... ExtraArgs>
class DataSourceImpl<PropertyT, MULTI_FIELD_PROPERTY, OutputSpaceT, VariadicTemplateTypesHolder<ExtraArgs...>>
//: public FieldProvider<boost::optional<typename PropertyAtSpace<PropertyT, OutputSpaceType>::ValueType>, OutputSpaceType, ExtraArgs...>    //inharistance only for change signal, not neccessery
{

    //shared_ptr<OutputSpaceType> destinationSpace;   //should be stored here? maybe only connection...

public:

    typedef OutputSpaceT OutputSpaceType;
    typedef typename PropertyT::EnumType EnumType;

    /**
     * Signal called when source has been changed.
     */
    boost::signals2::signal<void()> changed;

    /*shared_ptr<OutputSpaceType> getDestinationSpace() const { return destinationSpace; }

    virtual void setDestinationSpace(shared_ptr<OutputSpaceType>) { this->destinationSpace = destinationSpace; }*/

    /*
     * Check if this source can provide value for given point.
     * @param p point, in outer space coordinates
     * @return @c true only if this can provide data in given point @p p
     */
    //virtual bool canProvide(const Vec<OutputSpaceType::DIM, double>& p) const = 0;

    /// Type of property value in output space
    typedef typename PropertyAtSpace<PropertyT, OutputSpaceType>::ValueType ValueType;

    /*
     * Check if this source can provide value for given point and eventualy return this value.
     * @param p point (in outer space coordinates)
     * @param extra_args
     * @param method interpolation method to use
     * @return value in point @p, set only if this can provide data in given point @p p
     */
   // virtual boost::optional<ValueType> get(const Vec<OutputSpaceType::DIM, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const = 0;

   // virtual ValueT get(const Vec<OutputSpaceType::DIM, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const = 0;

    //virtual LazyData<boost::optional<ValueType>> operator()(const MeshD<OutputSpaceType::DIM>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const = 0;

    virtual std::function<boost::optional<ValueType>(std::size_t index)> operator()(EnumType num, const shared_ptr<const MeshD<OutputSpaceType::DIM>>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const = 0;

    virtual size_t size() const = 0;
    
    inline std::function<boost::optional<ValueType>(std::size_t index)> operator()(EnumType num, const shared_ptr<const MeshD<OutputSpaceType::DIM>>& dst_mesh, std::tuple<ExtraArgs...> extra_args, InterpolationMethod method) const {
        typedef std::tuple<ExtraArgs...> Tuple;
        return apply_tuple(dst_mesh, method, std::forward<Tuple>(extra_args), make_seq_indices<0, sizeof...(ExtraArgs)>{});
    }

private:
    template <typename T,  template <std::size_t...> class I, std::size_t... Indices>
    inline std::function<boost::optional<ValueType>(std::size_t index)> apply_tuple(const shared_ptr<const MeshD<OutputSpaceType::DIM>>& dst_mesh, InterpolationMethod method, T&& t, I<Indices...>) const {
      return this->operator()(dst_mesh, std::get<Indices>(std::forward<T>(t))..., method);
    }

};

template <typename PropertyT, typename OutputSpaceType>
using DataSource = DataSourceImpl<PropertyT, PropertyT::propertyType, OutputSpaceType, typename PropertyT::ExtraParams>;

// Hold reference to data source and destination mesh, base for LazyDataImpl returned by most DataSources
/*template <typename DataSourceType>
struct DataSourceDataImpl: public LazyDataImpl<boost::optional<typename DataSourceType::ValueType>> {

    const DataSourceType& data_src;
    const MeshD<DataSourceType::DIM>& dst_mesh;

    DataSourceDataImpl(const DataSourceType& data_src, const MeshD<DataSourceType::DIM>& dst_mesh): data_src(data_src), dst_mesh(dst_mesh) {}

    virtual std::size_t size() const override { return dst_mesh.size(); }

};*/

template <typename PropertyT, typename OutputSpaceType, typename InputSpaceType = OutputSpaceType, typename OutputGeomObj = OutputSpaceType, typename InputGeomObj = InputSpaceType>
struct DataSourceWithReceiver: public DataSource<PropertyT, OutputSpaceType> {

protected:
    //in, out obj can't be hold by shared_ptr, due to memory leak (circular reference)
    const InputGeomObj* inputObj;
    const OutputGeomObj* outputObj;
    boost::optional<PathHints> path;
    boost::signals2::connection geomConnectionIn;
    boost::signals2::connection geomConnectionOut;

public:
    ReceiverFor<PropertyT, InputSpaceType> in;

    DataSourceWithReceiver() {
        in.providerValueChanged.connect(
            [&] (ReceiverBase&, ReceiverBase::ChangeReason reason) {
                if (reason != ReceiverBase::ChangeReason::REASON_DELETE) this->changed();
            }
        );
    }

    ~DataSourceWithReceiver() {
         disconnect();
    }

    void disconnect() {
        geomConnectionIn.disconnect();
        geomConnectionOut.disconnect();
    }

    /**
     * This is called before request for data, but after setup inputObj, outputObj and path fields.
     * It can calculate trasnaltion and so needed for quick operator() calculation.
     */
    virtual void calcConnectionParameters() = 0;

    void setPath(const PathHints* path) {
        if (path)
            this->path = *path;
        else
            this->path = boost::optional<PathHints>();
    }

    const PathHints* getPath() const {
        return path ? &*path : nullptr;
    }

    void inOrOutWasChanged(GeometryObject::Event& e) {
        if (e.hasFlag(GeometryObject::Event::EVENT_DELETE)) disconnect(); else
        if (e.hasFlag(GeometryObject::Event::EVENT_RESIZE)) calcConnectionParameters();
    }

    void connect(InputGeomObj& inputObj, OutputGeomObj& outputObj, const PathHints* path = nullptr) {
        disconnect();
        this->setPath(path);
        this->inputObj = &inputObj;
        this->outputObj = &outputObj;
        geomConnectionOut = outputObj.changedConnectMethod(this, &DataSourceWithReceiver::inOrOutWasChanged);
        geomConnectionIn = inputObj.changedConnectMethod(this, &DataSourceWithReceiver::inOrOutWasChanged);
        calcConnectionParameters();
    }
};

template <typename PropertyT, typename OutputSpaceType, typename InputSpaceType = OutputSpaceType, typename OutputGeomObj = OutputSpaceType, typename InputGeomObj = InputSpaceType>
struct InnerDataSource: public DataSourceWithReceiver<PropertyT, OutputSpaceType, InputSpaceType, OutputGeomObj, InputGeomObj> {

    typedef typename Primitive<OutputSpaceType::DIM>::Box OutBox;
    typedef Vec<OutputSpaceType::DIM, double> OutVec;

    struct Region {

        /// Input bounding-box in output geometry.
        OutBox inGeomBB;

        /// Translation to input object (before eventual space reduction).
        OutVec inTranslation;

        Region(const OutBox& inGeomBB, const OutVec& inTranslation)
            : inGeomBB(inGeomBB), inTranslation(inTranslation) {}

    };

    std::vector<Region> regions;

    const Region* findRegion(const OutVec& p) const {
        for (const Region& r: regions)
            if (r.inGeomBB.contains(p)) return &r;
        return nullptr;
    }

    const std::size_t findRegionIndex(const OutVec& p) const {
        for (std::size_t i = 0; i < regions.size(); ++i)
            if (regions[i].inGeomBB.contains(p)) return i;
        return regions.size();
    }

    /**
     * Find region that has @p p inside bouding-box and fulfill predicate @p pred.
     */
    template <typename Predicate>
    const Region* findRegion(const OutVec& p, Predicate pred) const {
        for (const Region& r: regions)
            if (r.inGeomBB.contains(p) && pred(r))
                return &r;
        return nullptr;
    }

    template <typename Predicate>
    const std::size_t findRegionIndex(const OutVec& p, Predicate pred) const {
        for (std::size_t i = 0; i < regions.size(); ++i)
            if (regions[i].inGeomBB.contains(p) && pred(regions[i])) return i;
        return regions.size();
    }

    virtual void calcConnectionParameters() override {
        regions.clear();
        std::vector<OutVec> pos = this->outputObj->getObjectPositions(*this->inputObj, this->getPath());
        for (auto& p: pos) {
            if (std::isnan(p.c0))
                throw plask::Exception("Filter error: the place of some source geometry inside a destination geometry can't be described by translation.\nThis can be caused by flip or mirror on the path from the source to the destination.");
        }
        std::vector<OutBox> bb = this->outputObj->getObjectBoundingBoxes(*this->inputObj, this->getPath());
        for (std::size_t i = 0; i < pos.size(); ++i)
            regions.emplace_back(bb[i], pos[i]);
    }

};

/*template <typename DataSourceType, typename... ExtraArgs>
struct InnerLazySourceImpl {    //template for base class

    std::vector<LazyData<typename DataSourceType::ValueType>> dataForRegion;

    const DataSourceType& source;

    const shared_ptr<const MeshD<DataSourceType::OutputSpaceType::DIM>> dst_mesh;

    std::tuple<ExtraArgs...> extra_args;

    InterpolationMethod method;

    InnerLazySourceImpl(const DataSourceType& source, const shared_ptr<const MeshD<DataSourceType::OutputSpaceType::DIM>>& dst_mesh,
                   ExtraArgs... extra_args, InterpolationMethod method)
        : dataForRegion(source.regions.size()), source(source), dst_mesh(dst_mesh), extra_args(extra_args...), method(method)
    {}
};*/

/// Data source in which input object is outer and contains output object.
template <typename PropertyT, typename OutputSpaceType, typename InputSpaceType = OutputSpaceType, typename OutputGeomObj = OutputSpaceType, typename InputGeomObj = InputSpaceType>
struct OuterDataSource: public DataSourceWithReceiver<PropertyT, OutputSpaceType, InputSpaceType, OutputGeomObj, InputGeomObj> {

    Vec<InputGeomObj::DIM, double> inTranslation;

    virtual void calcConnectionParameters() override {
        std::vector<Vec<InputGeomObj::DIM, double>> pos = this->inputObj->getObjectPositions(*this->outputObj, this->getPath());
        if (pos.size() != 1) throw Exception("Inner output geometry object has not unambiguous position in outer input geometry object.");
        inTranslation = pos[0];
    }

};

/// Don't use this directly, use ConstDataSource instead.
template <typename PropertyT, PropertyType propertyType, typename OutputSpaceType, typename VariadicTemplateTypesHolder>
struct ConstDataSourceImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "filter data sources can't be use with single value properties (it can be use only with fields properties)");
};

template <typename PropertyT, typename OutputSpaceType, typename... ExtraArgs>
struct ConstDataSourceImpl<PropertyT, FIELD_PROPERTY, OutputSpaceType, VariadicTemplateTypesHolder<ExtraArgs...>>
        : public DataSource<PropertyT, OutputSpaceType> {

public:

    /// Type of property value in output space
    typedef typename PropertyAtSpace<PropertyT, OutputSpaceType>::ValueType ValueType;

    ValueType value;

    ConstDataSourceImpl(const ValueType& value): value(value) {}

    std::function<boost::optional<ValueType>(std::size_t index)> operator()(const shared_ptr<const MeshD<OutputSpaceType::DIM>>&, ExtraArgs..., InterpolationMethod) const override {
        return [=](std::size_t) { return value; };
    }

};

template <typename PropertyT, typename OutputSpaceType>
using ConstDataSource = ConstDataSourceImpl<PropertyT, PropertyT::propertyType, OutputSpaceType, typename PropertyT::ExtraParams>;

}   // plask

#endif // PLASK__FILTER__BASE_H
