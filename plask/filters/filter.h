#ifndef PLASK__FILTER_H
#define PLASK__FILTER_H

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

template <typename PropertyT, typename OutputSpaceType, typename... ExtraArgs>
class DataSourceImpl<PropertyT, FIELD_PROPERTY, OutputSpaceType, VariadicTemplateTypesHolder<ExtraArgs...>>: public ProviderFor<PropertyT> {

    //shared_ptr<OutputSpaceType> destinationSpace;   //should be stored here? maybe only connection...

public:

    /*shared_ptr<OutputSpaceType> getDestinationSpace() const { return destinationSpace; }

    virtual void setDestinationSpace(shared_ptr<OutputSpaceType>) { this->destinationSpace = destinationSpace; }*/

    /**
     * Check if this source can provide value for given point.
     * @param p point, in outer space coordinates
     * @return @c true only if this can provide data in given point @p p
     */
    virtual bool canProvide(const Vec<OutputSpaceType::DIMS, double>& p) const = 0;

   // virtual ValueT get(const Vec<OutputSpaceType::DIMS, double>& p, ExtraArgs... extra_args, InterpolationMethod method) const = 0;

   /* virtual DataVector<const typename PropertyT::ValueType> operator()(const MeshD<SpaceT::DIMS>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const {
        DataVector<ValueT> result(dst_mesh.size());
        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = get(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
        return result;
    }*/

};

template <typename PropertyT, typename OutputSpaceType>
using DataSource = DataSourceImpl<PropertyT, PropertyT::propertyType, OutputSpaceType, typename PropertyT::ExtraParams>;

template <typename PropertyT, typename OutputSpaceType, typename InputSpaceType = OutputSpaceType>
struct DataSourceWithReceiver: public DataSource<PropertyT, OutputSpaceType> {

    ReceiverFor<PropertyT, InputSpaceType> in;

    //shared_ptr<InputSpaceType> inSpace;   //or only connection?

    DataSourceWithReceiver() {
        in.providerValueChanged.connect([&] (Provider::Listener&) { this->fireChanged(); });
    }

};

/// Don't use this directly, use StandardFilter instead.
template <typename PropertyT, PropertyType propertyType, typename OutputSpaceType, typename VariadicTemplateTypesHolder>
struct CombineDataSourceImpl {
    static_assert(propertyType != SINGLE_VALUE_PROPERTY, "space change filter data sources can't be use with single value properties (it can be use only with fields properties)");
};

template <typename PropertyT, typename OutputSpaceType, typename... ExtraArgs>
struct CombineDataSourceImpl< PropertyT, FIELD_PROPERTY, OutputSpaceType, VariadicTemplateTypesHolder<ExtraArgs...> >
    : public DataSource<PropertyT, OutputSpaceType>
{
    //one outer source (input)
    //vector if inner sources (inputs)
    //one output (provider)

    typedef DataSource<PropertyT, OutputSpaceType> DataSourceT;
    typedef std::unique_ptr<DataSourceT> DataSourceTPtr;

    std::vector<DataSourceTPtr> innerSources;

    DataSourceTPtr outerSource;

    virtual DataVector<const typename PropertyT::ValueType> operator()(const MeshD<OutputSpaceType::DIMS>& dst_mesh, ExtraArgs... extra_args, InterpolationMethod method) const {
        //iterate over inner sources
        //if inner sources don't provide data, use outer source
        /*DataVector<ValueT> result(dst_mesh.size());
            for (std::size_t i = 0; i < result.size(); ++i)
                result[i] = get(dst_mesh, std::forward<ExtraArgs>(extra_args)..., method);
            return result;*/
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

struct PointsOnLineMesh: public MeshD<3> {

    std::size_t lastPointNr;

    Vec<3, double> begin;

    double longSize;

    PointsOnLineMesh() = default;

    PointsOnLineMesh(Vec<3, double> begin, double lonSize, std::size_t pointsCount)
        : lastPointNr(pointsCount-1), begin(begin), longSize(lonSize) {}

    virtual Vec<3, double> at(std::size_t index) const override {
        Vec<3, double> ans = begin;
        ans.lon() += longSize * index / lastPointNr;
        return ans;
    }

    virtual std::size_t size() const override {
        return lastPointNr + 1;
    }

};

struct Mesh2DCartTo3D: public MeshD<3> {

    const MeshD<2>& sourceMesh;

    double lon;

    Mesh2DCartTo3D(const MeshD<2>& sourceMesh, double lon)
        : sourceMesh(sourceMesh), lon(lon) {}

    virtual Vec<3, double> at(std::size_t index) const override {
        return vec(sourceMesh.at(index), lon);
    }

    virtual std::size_t size() const override {
        return sourceMesh.size();
    }
};

struct Cartesian2Dto3DEnviroment {

    /// Place of extrusion inside outer 3D geometry, calculated using geometries.
    Vec<3, double> translation;

    /// Length of line in 3D geometry. This line is calculated for each point in inner 2D space. Length from extrusion.
    double lonSize;

    Cartesian2Dto3DEnviroment() {}

    Cartesian2Dto3DEnviroment(Vec<3, double> translation, double lonSize)
        : translation(translation), lonSize(lonSize) {}

    void setVertTran(Vec<3,double>& result, const Vec<2,double>& v) const {
        result.tran() = translation.tran() + v.tran();
        result.vert() = translation.vert() + v.vert();
    }

    /**
     * Each point in 2D geometry is extruded to line segment in 3D geometry.
     *
     * If @c b is begin of this line segment, than each point inside this segment has coordinate
     * which can be calulate by adding to @c b.lon() number in range [0.0, lonSize].
     * @param v point in 2D geometry coordinates
     * @return begin of line segment, in geometry 3D coordinates
     */
    Vec<3,double> lineBegin(const Vec<2,double>& v) const {
        Vec<3, double> result = translation;
        result.tran() += v.tran();
        result.vert() += v.vert();
        return result;
    }

    /**
     * Each point in 2D geometry is extruded to line segment in 3D geometry.
     * @param v point in 2D geometry coordinates
     * @return end of line segment, in geometry 3D coordinates
     * @see lineBegin(const Vec<2,double>&)
     */
    Vec<3,double> lineEnd(const Vec<2,double>& v) const {
        Vec<3, double> result = translation;
        result.tran() += v.tran();
        result.vert() += v.vert();
        result.lon() += lonSize;
        return result;
    }

    /**
     * Each point in 2D geometry is extruded to line in segment 3D geometry.
     * @param v point in 2D geometry coordinates
     * @param at position in line, from 0.0 to 1.0: 0.0 for line begin, 1.0 for line end
     * @return point at line segment, in geometry 3D coordinates
     * @see lineBegin(const Vec<2,double>&)
     */
    Vec<3,double> lineAt(const Vec<2,double>& v, double at) const {
        Vec<3, double> result = translation;
        result.tran() += v.tran();
        result.vert() += v.vert();
        result.lon() += lonSize * at;
        return result;
    }
};

/// Don't use this directly, use ChangeSpaceCartesian2Dto3D instead.
template <typename PropertyT, typename VariadicTemplateTypesHolder>
struct ChangeSpaceCartesian2Dto3DImpl: public StandardFilter<PropertyT, Geometry3D, Geometry2DCartesian> {};

/// Don't use this directly, use ChangeSpaceCartesian2Dto3D instead.
template <typename PropertyT, typename... ExtraParams>
class ChangeSpaceCartesian2Dto3DImpl<PropertyT, VariadicTemplateTypesHolder<ExtraParams...> >
        : public StandardFilter<PropertyT, Geometry3D, Geometry2DCartesian> {

    boost::signals2::connection geomConnection;
    
public:

    // type of ChangeSpaceCartesian2Dto3D logic, it calculates and returns values at requested_points using given data_source and enviroment
    /*typedef std::function<
        DataVector<const typename PropertyT::ValueType>(
            const MeshD<2>& requested_points,
            DataSource<3, const typename PropertyT::DataType> data_source,
            const Cartesian2Dto3DEnviroment& enviroment
        )
    > Logic;

    Logic logic;*/  //good, but unused now, allow to custom avarage logic, to enable when need

    /// Points count for avarage function
    std::size_t pointsCount;    //eventualy should be moved to logic
    
    Cartesian2Dto3DEnviroment enviroment;
    
    void getParameters(shared_ptr<const Geometry3D> outerInGeometry, shared_ptr<const Extrusion> innerOutGeometry, const PathHints* path = nullptr) {
        std::vector< Vec<3, double> > pos = outerInGeometry->getObjectPositions(innerOutGeometry, path);
        if (pos.size() != 1) throw Exception("ChangeSpaceCartesian2Dto3D: innerOutGeometry has no unambiguous position in outerInGeometry.");
        enviroment.translation = pos[0];
        enviroment.lonSize = innerOutGeometry->getLength();
    }
    
    ChangeSpaceCartesian2Dto3DImpl(const Vec<3, double>& translation, double lonSize, std::size_t pointsCount = 10)
        : StandardFilter<PropertyT, Geometry3D, Geometry2DCartesian>("ChangeSpaceCartesian2Dto3DFilter"), pointsCount(pointsCount), enviroment(translation, lonSize) {}
    
    ChangeSpaceCartesian2Dto3DImpl(shared_ptr<Geometry3D> outerInGeometry, shared_ptr<const Extrusion> innerOutGeometry, const PathHints* path = nullptr, std::size_t pointsCount = 10)
        : StandardFilter<PropertyT, Geometry3D, Geometry2DCartesian>("ChangeSpaceCartesian2Dto3DFilter"), pointsCount(pointsCount)
    {
        getParameters(outerInGeometry, innerOutGeometry, path);
        if (path) {
            PathHints pathCopy = *path;
//            geomConnection = outerInGeometry->changed.connect([=](GeometryObject::Event& e) {
//                    if (e.hasFlag(GeometryObject::Event::RESIZE)) getParameters(outerInGeometry, innerOutGeometry, &pathCopy);
//            });
        } //else
//            geomConnection = outerInGeometry->changed.connect([=](GeometryObject::Event& e) {
//                    if (e.hasFlag(GeometryObject::Event::RESIZE)) getParameters(outerInGeometry, innerOutGeometry);
//            });   //Windows GCC 4.7.1 not compile this
    }
    
    ~ChangeSpaceCartesian2Dto3DImpl() {
        geomConnection.disconnect();
    }

    virtual DataVector<const typename PropertyT::ValueType> apply(const MeshD<2>& requested_points, ExtraParams... extra_args, InterpolationMethod method = DEFAULT_INTERPOLATION) {
        /*return logic(
            requested_points,
            [&, extra_args...] (const MeshD<3>& dst_mesh) -> DataVector<const typename PropertyT::DataType> {
                return in(dst_mesh, extra_args..., method);
            },
            enviroment
        );    //good code but not supported by GCC 4.7
        // workaround:
        auto t = std::make_tuple(std::ref(extra_args)...);
        return logic(
                requested_points,
                [&] (const MeshD<3>& dst_mesh) -> DataVector<const typename PropertyT::DataType> {
                    return in(dst_mesh, t, method);
                },
                enviroment
            );*/
        if (pointsCount == 1) {
            return this->in(
                        Mesh2DCartTo3D(requested_points, enviroment.translation.lon() + enviroment.lonSize * 0.5),
                        std::forward<ExtraParams>(extra_args)...,
                        method
                    );
        } else {
            DataVector<typename PropertyT::ValueType> result(requested_points.size());
            PointsOnLineMesh lineMesh;
            const double d = enviroment.lonSize / result.size();
            lineMesh.lastPointNr = result.size() - 1;
            lineMesh.longSize = enviroment.lonSize - d;
            lineMesh.begin.lon() = enviroment.translation.lon() + d * 0.5;
            for (std::size_t src_point_nr = 0; src_point_nr < result.size(); ++src_point_nr) {
                enviroment.setVertTran(lineMesh.begin, requested_points[src_point_nr]);
                result[src_point_nr] =
                        avarage(this->in(
                            lineMesh,
                            std::forward<ExtraParams>(extra_args)...,
                            method
                        ));
            }
            return result;
        }
    }
};

/**
 * Filter which recalculate field from 3D geometry to 2D cartesian geometry included in this 3D geometry.
 */
template <typename PropertyT>
using ChangeSpaceCartesian2Dto3D = ChangeSpaceCartesian2Dto3DImpl<PropertyT, typename PropertyT::ExtraParams>;



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
