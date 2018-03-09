#ifndef PLASK__MESH_H
#define PLASK__MESH_H

/** @file
This file contains base classes for meshes.
@see @ref meshes
*/

/** @page meshes Meshes
@section meshes_about About meshes
The mesh represents (ordered) set of points in 2D or 3D space. All meshes in PLaSK implements (inherits from)
instantiation of plask::Mesh template interface.

Typically, there is some data associated with points in mesh.
In PLaSK, all this data is not stored in the mesh class, hence they must be stored separately.
As the points in the mesh are ordered and each one have unique index in a range from <code>0</code>
to <code>plask::Mesh::size()-1</code>,
you can store data in any indexed structure, like an array (1D) or std::vector (which is recommended),
storing the data value for the i-th point in the mesh under the i-th index.

@see @ref interpolation @ref boundaries

@section meshes_write How to implement a new mesh?
Typical approaches to implementing new types of meshes:
- @ref meshes_write_direct (this approach is the most flexible).

@see @ref interpolation_write @ref boundaries_impl

@subsection meshes_write_direct Direct implementation of plask::MeshD\<DIM\>
To implement a new mesh directly you have to write class inherited from the \c plask::MeshD\<DIM\>, where DIM (is equal 2 or 3) is a number of dimension of space your mesh is defined over.

You are required to:
- implement the @ref plask::Mesh::size size and @ref plask::MeshD::at at methods which allow to access to mesh points;
- implement the @ref plask::MeshD::writeXML method, which writes the mesh to XML;
- write and the reading function which reads the mesh from XML.

Example implementation of singleton mesh (mesh which represent set with only one point in 3D space):
@code
struct OnePoint3DMesh: public plask::MeshD<3> {

    // Held point:
    plask::Vec<3, double> point;

    OnePoint3DMesh(const plask::Vec<3, double>& point)
    : point(point) {}

    // plask::MeshD<3> methods implementation:

    virtual std::size_t size() const override {
        return 1;
    }

    virtual plask::Vec<3, double> at(std::size_t index) const override {
        return point;
    }

    virtual void writeXML(XMLElement& object) const override {
        object.attr("type", "point3d"); // this is required attribute for the provided object
        object.addTag("point")
               .attr("c0", point.c0)
               .attr("c1", point.c1)
               .attr("c2", point.c2);   // store this point coordinates in attributes of the tag <point>
    }

};

// Now write reading function (when it is called, the current tag is the <mesh> tag):

static shared_ptr<Mesh> readOnePoint3DMesh(plask::XMLReader& reader) {
    reader.requireTag("point");
    double c0 = reader.requireAttribute<double>("c0");
    double c1 = reader.requireAttribute<double>("c1");
    double c2 = reader.requireAttribute<double>("c2");
    reader.requireTagEnd();   // this is necessary to make sure the tag <point> is closed
    // Now create the mesh into a shared pointer and return it:
    return plask::make_shared<OnePoint3DMesh>(plask::Vec<3,double>(c0, c1, c2));
}

// Declare global variable of type RegisterMeshReader in order to register the reader:
//   the first argument must be the same string which has been written into 'type' attribute in OnePoint3DMesh::writeXML() method,
//   the second one is the address of your reading function,
//   variable name does not matter.

static RegisterMeshReader onepoint3dmesh_reader("point3d", &readOnePoint3DMesh);

@endcode
You should also implement interpolation algorithms for your mesh, see @ref interpolation_write for more details.
*/

#include <map>

#include <plask/config.h>
#include <plask/memory.h>

#include "../vec.h"
#include "../geometry/object.h"
#include "../utils/iterators.h"
#include "../utils/cache.h"
#include "../utils/xml.h"

#include <boost/signals2.hpp>
#include "../utils/event.h"

namespace plask {

/**
 * Base class for all the meshes.
 * Mesh represent a set of points in 2D or 3D space and:
 * - knows number of points,
 * - allows for iterate over this points,
 * - can calculate interpolated value for given destination points, source values, and the interpolation method,
 * - inform about self changes.
 *
 * @see @ref meshes
 */

/// Common base for meshes and generators
struct PLASK_API MeshBase {
    virtual ~MeshBase() {}
};

struct PLASK_API Mesh: public Printable, MeshBase {


    /**
     * Store information about event connected with mesh.
     *
     * Subclasses of this can contains additional information about specific type of event.
     */
    struct Event: public EventWithSourceAndFlags<Mesh> {

        /// Event flags (which describe event properties).
        enum Flags {
            EVENT_DELETE = 1,             ///< is deleted
            EVENT_RESIZE = 1<<1,          ///< size could be changed (points added or deleted)
            EVENT_USER_DEFINED = 1<<2     ///< user-defined flags could have ids: EVENT_USER_DEFINED, EVENT_USER_DEFINED<<1, EVENT_USER_DEFINED<<2, ...
        };

        /**
         * Check if given @p flag is set.
         * @param flag flag to check
         * @return @c true only if @p flag is set
         */
        bool hasFlag(Flags flag) const { return hasAnyFlag(flag); }

        /**
         * Check if EVENT_DELETE flag is set, which mean that source of event is being deleted.
         * @return @c true only if EVENT_DELETE flag is set
         */
        bool isDelete() const { return hasFlag(EVENT_DELETE); }

        /**
         * Check if EVENT_RESIZE flag is set, which mean that source of event could have been resized.
         * @return @c true only if EVENT_RESIZE flag is set
         */
        bool isResize() const { return hasFlag(EVENT_RESIZE); }

        /**
         * Construct the event.
         * @param source source of event
         * @param flags flags which describes event's properties
         */
        explicit Event(Mesh* source, unsigned char flags = 0): EventWithSourceAndFlags<Mesh>(source, flags) {}
    };

    /// Changed signal, fired when mesh was changed.
    boost::signals2::signal<void(Event&)> changed;

    /**
     * Connect a method to changed signal.
     * @param obj, method slot to connect, object and it's method
     * @param at specifies where the slot should be connected:
     *  - boost::signals2::at_front indicates that the slot will be connected at the front of the list or group of slots
     *  - boost::signals2::at_back (default) indicates that the slot will be connected at the back of the list or group of slots
     */
    template <typename ClassT, typename methodT>
    boost::signals2::connection changedConnectMethod(ClassT* obj, methodT method, boost::signals2::connect_position at = boost::signals2::at_back) {
        return changed.connect(boost::bind(method, obj, _1), at);
    }

    template <typename ClassT, typename methodT>
    void changedDisconnectMethod(ClassT* obj, methodT method) {
        changed.disconnect(boost::bind(method, obj, _1));
    }

    /**
     * Call changed with this as event source.
     * @param event_constructor_params_without_source parameters for event constructor (without first - source)
     */
    template<typename EventT = Event, typename ...Args>
    void fireChanged(Args&&... event_constructor_params_without_source) {
        EventT evt(this, std::forward<Args>(event_constructor_params_without_source)...);
        onChange(evt);
        changed(evt);
    }

    /// This method is called when the mesh is resized
    void fireResized() { fireChanged(Event::EVENT_RESIZE); }

    /// @return number of points in mesh
    virtual std::size_t size() const = 0;

    /// @return @c true only if mesh is empty (there are no points in mesh)
    virtual bool empty() const { return size() == 0; }

    /**
     * Write mesh to XML
     * \param object XML object to write to
     */
    virtual void writeXML(XMLElement& object) const;

    virtual ~Mesh() { fireChanged(Event::EVENT_DELETE); }

  protected:

    /**
     * This method is called when the mesh is changed, just before changed signal.
     * \param evt triggering event
     */
    virtual void onChange(const Event& evt);

};

/**
 * Base class for all meshes defined for specified number of dimensions.
 */
template <int dimension>
struct PLASK_API MeshD: public Mesh {

    /// Number of dimensions
    enum { DIM = dimension };

    /// Type of vector representing coordinates in local space
    typedef typename Primitive<DIM>::DVec LocalCoords;

    /**
     * Get point with given mesh index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    virtual LocalCoords at(std::size_t index) const = 0;

    /**
     * Get point with given mesh index.
     * @param index index of point, from 0 to size()-1
     * @return point with given @p index
     */
    const LocalCoords operator[](std::size_t index) const { return at(index); }

    /**
     * Random access iterator type which allow iterate over all points in this mesh, in order appointed by operator[].
     * This iterator type is indexed, which means that it have (read-write) index field equal to 0 for begin() and growing up to size() for end().
     */
    typedef IndexedIterator< const MeshD<dimension>, LocalCoords > const_iterator;
    typedef const_iterator iterator;
    typedef const_iterator Iterator;

    /// @return iterator referring to the first point in this mesh
    const_iterator begin() const { return const_iterator(this, 0); }

    /// @return iterator referring to the past-the-end point in this mesh
    const_iterator end() const { return const_iterator(this, this->size()); }

    /**
     * Initialize this to be the same as @p to_copy but don't copy any changes observer.
     * @param to_copy object to copy
     */
    MeshD(const MeshD& PLASK_UNUSED(to_copy)) {}

    MeshD() {}

    /**
     * Set this to be the same as @p to_copy but doesn't changed changes observer.
     * @param to_copy object to copy
     */
    MeshD& operator=(const MeshD& PLASK_UNUSED(to_copy)) { return *this; }

    /**
     * Check if this mesh and @p to_compare represent the same sequence of points (have exactly the same points in the same order).
     * @param to_compare mesh to compare
     * @return @p to_compare represent the same sequence of points as this
     */
    bool operator==(const MeshD& to_compare) const;

    /**
     * Check if this mesh and @p to_compare represent different sequences of points.
     * @param to_compare mesh to compare
     * @return @c true only if this mesh and @p to_compare represent different sequences of points
     */
    bool operator!=(const MeshD& to_compare) const { return ! (*this == to_compare); }

    void print(std::ostream& out) const override;

};

PLASK_API_EXTERN_TEMPLATE_STRUCT(MeshD<1>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(MeshD<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(MeshD<3>)

/** Base template for rectangular mesh of any dimension */
template <int dim> class RectangularMesh {};

/** Base class for every mesh generator */
class PLASK_API MeshGenerator: public MeshBase {
  public:

    /// Mesh generator event.
    typedef EventWithSourceAndFlags<MeshGenerator> Event;

    /// Changed signal, fired when space was changed.
    boost::signals2::signal<void(Event&)> changed;

    /**
     * Connect a method to changed signal.
     * @param obj, method slot to connect, object and it's method
     * @param at specifies where the slot should be connected:
     *  - boost::signals2::at_front indicates that the slot will be connected at the front of the list or group of slots
     *  - boost::signals2::at_back (default) indicates that the slot will be connected at the back of the list or group of slots
     */
    template <typename ClassT, typename methodT>
    void changedConnectMethod(ClassT* obj, methodT method, boost::signals2::connect_position at = boost::signals2::at_back) {
        changed.connect(boost::bind(method, obj, _1), at);
    }

    template <typename ClassT, typename methodT>
    void changedDisconnectMethod(ClassT* obj, methodT method) {
        changed.disconnect(boost::bind(method, obj, _1));
    }

    /**
     * Call changed with this as event source.
     * @param event_constructor_params_without_source parameters for event constructor (without first - source)
     */
    template<typename EventT = Event, typename ...Args>
    void fireChanged(Args&&... event_constructor_params_without_source) {
        EventT evt(this, std::forward<Args>(event_constructor_params_without_source)...);
        onChange(evt);
        changed(evt);
    }

    virtual ~MeshGenerator() {}

  protected:

    /**
     * This method is called when the generator is changed, just before changed signal.
     * \param evt triggering event
     */
    virtual void onChange(const Event& evt);

};

/** Base class for specific mesh generator */
template <int MESH_DIM>
class PLASK_API MeshGeneratorD: public MeshGenerator
{
  public:
      /// Type of the generated mesh
      typedef MeshD<MESH_DIM> MeshType;

      /// Number of geometry dimensions
      enum { DIM = (MESH_DIM < 2) ? 2 : MESH_DIM };

  protected:
    WeakCache<GeometryObject, MeshType, CacheRemoveOnEachChange> cache;

    void onChange(const Event&) override { clearCache(); }

    template <typename RequiredType>
    static shared_ptr<RequiredType> cast(const shared_ptr<MeshType>& res) {
        auto finall_res = dynamic_pointer_cast<RequiredType>(res);
        if (res && !finall_res) throw Exception("Wrong type of generated {0}D mesh.", MESH_DIM);
        return finall_res;
    }

  public:

	typedef shared_ptr<GeometryObjectD<DIM>> GeometryPtr;

    /**
     * Generate new mesh
     * \param geometry on which the mesh should be generated
     * \return new generated mesh
     */
    virtual shared_ptr<MeshType> generate(const GeometryPtr& geometry) = 0;

    /**
     * Clear the cache of generated meshes.
     * This method should be called each time any parameter of generator is changed
     */
    inline void clearCache() {
        cache.clear();
    }

    /// Get generated mesh if it is cached or create a new one
    shared_ptr<MeshType> operator()(const GeometryPtr& geometry);

    template <typename RequiredType>
    shared_ptr<RequiredType> get(const shared_ptr<GeometryObjectD<DIM>>& geometry) {
        return cast<RequiredType> ( this->operator ()(geometry) );
    }

    template <typename RequiredType>
    shared_ptr<RequiredType> generate_t(const shared_ptr<GeometryObjectD<DIM>>& geometry) {
        return cast<RequiredType> ( this->generate(geometry) );
    }

};

PLASK_API_EXTERN_TEMPLATE_CLASS(MeshGeneratorD<1>)
PLASK_API_EXTERN_TEMPLATE_CLASS(MeshGeneratorD<2>)
PLASK_API_EXTERN_TEMPLATE_CLASS(MeshGeneratorD<3>)

/**
 * Helper which call stores mesh reader when constructed.
 * Each mesh can create one global instance of this class to its reader.
 */
struct PLASK_API RegisterMeshReader {
    typedef std::function<shared_ptr<Mesh>(XMLReader&)> ReadingFunction;
    RegisterMeshReader(const std::string& tag_name, ReadingFunction fun);
    static std::map<std::string, ReadingFunction>& getReaders();
    static ReadingFunction getReader(const std::string& name);
};

struct Manager;

/**
 * Helper which call stores mesh reader when constructed.
 * Each mesh can create one global instance of this class to its reader.
 */
struct PLASK_API RegisterMeshGeneratorReader {
    typedef std::function<shared_ptr<MeshGenerator>(XMLReader&, const Manager&)> ReadingFunction;
    RegisterMeshGeneratorReader(const std::string& tag_name, ReadingFunction fun);
    static std::map<std::string, ReadingFunction>& getReaders();
    static ReadingFunction getReader(const std::string& name);
};


} // namespace plask

#endif  //PLASK__MESH_H
