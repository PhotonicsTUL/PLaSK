#ifndef PLASK__MESH_H
#define PLASK__MESH_H

/** @file
This file includes base classes for meshes.
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
There are two typical approaches to implementing new types of meshes:
- @ref meshes_write_adapters "using adapters" (this approach is recommended),
- @ref meshes_write_direct "direct".

@see @ref interpolation_write @ref boundaries_impl

@subsection meshes_write_adapters Using adapters to generate plask::Mesh implementation
You can specialize adapter template to generate class which inheriting from plask::Mesh instantiation.

To do this, you have to implement internal mesh representation class first.

Typically internal mesh interface:
- represents set of point in the same space as parent mesh;
- allow for faster calculation than generic mesh interface, and often has more futures (methods);
- can have different types (there are no common base class for internal interfaces).

In most cases, mesh adapter has @c internal field which is internal mesh interface and use @c internal field methods to implement itself methods (especially, abstract methods of plask::Mesh).

Your class must fulfill adapter templates requirements (it is one of adapter template parameters),
and also can have extra methods for your internal use (for calculation).

Adapter templates currently available in PLaSK (see its description for more details and examples):
- plask::SimpleMeshAdapter

@subsection meshes_write_direct Direct implementation of plask::MeshD\<DIM\>
To implement a new mesh directly you have to write class inherited from the \c plask::MeshD\<DIM\>, where DIM (is equal 2 or 3) is a number of dimension of space your mesh is defined over.

You are required to:
- implement the @ref plask::Mesh::size size method;
- implement the iterator over the mesh points, which required to:
  - writing class inherited from plask::Mesh::IteratorImpl (and implement all its abstract methods),
  - writing @ref plask::MeshD::begin "begin()" and @ref plask::MeshD::end "end()" methods, typically this methods only returns:
    @code plask::Mesh::Iterator(new YourIteratorImpl(...)) @endcode
  - see also: MeshIteratorWrapperImpl and makeMeshIterator
- implement the \ref plask::MeshD::writeXML method, which writes the mesh to XML
- write and register the reading function which reads the mesh from XML

Example implementation of singleton mesh (mesh which represent set with only one point in 3D space):
@code
struct OnePoint3DMesh: public plask::MeshD<3> {

    // Held point:
    plask::Vec<3, double> point;

    OnePoint3DMesh(const plask::Vec<3, double>& point)
    : point(point) {}

    // Iterator:
    struct IteratorImpl: public MeshD<plask::space::Cartesian3D>::IteratorImpl {

        // Pointer to mesh or is equal to nullptr for end iterator
        const OnePoint3DMesh* mesh_ptr;

        // mesh == nullptr for end iterator
        IteratorImpl(const OnePoint3DMesh* mesh)
        : mesh_ptr(mesh) {}

        virtual const plask::Vec<3, double> dereference() const {
            return mesh_ptr->point;
        }

        virtual void increment() {
            mesh_ptr = nullptr; // we iterate only over one point, so next state is end
        }

        virtual bool equal(const typename MeshD<plask::space::Cartesian3D>::IteratorImpl& other) const {
            return mesh_ptr == static_cast<const IteratorImpl&>(other).mesh_ptr;
        }

        virtual IteratorImpl* clone() const {
            return new IteratorImpl(mesh_ptr);
        }

        virtual std::size_t getIndex() const {
            return 0;
        }

    };

    // plask::MeshD<3> methods implementation:

    virtual std::size_t size() const {
        return 1;
    }

    virtual typename MeshD<plask::space::Cartesian3D>::Iterator begin() const {
        return MeshD<3>::Iterator(new IteratorImpl(this));
    }

    virtual typename MeshD<plask::space::Cartesian3D>::Iterator end() const {
        return MeshD<3>::Iterator(new IteratorImpl(nullptr));
    }

    virtual void writeXML(XMLElement& element) const;
        element.attr("type", "point3d"); // these is required attribute for the provided element
        element.addTag("point")
               .attr("c0", point.c0)
               .attr("c1", point.c1)
               .attr("c2", point.c2)     // store this point coordinates in attributes of the tag <point>
        ;
    }

};

// Now write reading function (when it is called, the current tag is the <mesh> tag):

static shared_ptr<Mesh> readOnePoint3DMesh(plask::XMLReader& reader) {
    reader.requireTag("point");
    double c0 = reader.requireAttribute<double>("c0");
    double c1 = reader.requireAttribute<double>("c1");
    double c2 = reader.requireAttribute<double>("c1");
    reader.requireTagEnd();   // this is necessary to make sure the tag <point> is closed
    // Now create the mesh into a shared pointer and return it:
    return make_shared<OnePoint3DMesh>(plask::Vec<3,double>(c0, c1, c2));
}

// Declare global variable of type RegisterMeshReader in order to register the reader:
//   the first argument must be the same string which was written in 'type' attribute in OnePoint3DMesh::writeXML() method
//   the second one is the address of your reading function
//   variable name does not matter

static RegisterMeshReader onepoint3dmesh_reader("point3d", &readOnePoint3DMesh);

@endcode
You should also implement interpolation algorithms for your mesh, see @ref interpolation_write for more details.
*/

#include <map>

#include <plask/config.h>
#include <plask/memory.h>

#include "../vec.h"
#include "../geometry/element.h"
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

struct Mesh {
    /// @return number of points in mesh
    virtual std::size_t size() const = 0;

    /**
     * Write mesh to XML
     * \param element XML element to write to
     */
    virtual void writeXML(XMLElement& element) const {
        throw NotImplemented("Mesh::writeXML()");
    }

    virtual ~Mesh() {}
};

/**
 * Base class for all meshes defined for specified number of dimensions.
 */
template <int dimension>
struct MeshD: public Mesh {

    /// Number of dimensions
    static const int dim = dimension;

    ///@return true only if there are no points in mesh
    bool empty() const { return size() == 0; }

    /// Type of vector representing coordinates in local space
    typedef Vec<dim, double> LocalCoords;

    /// Base class for mesh iterator implementation.
    typedef PolymorphicForwardIteratorWithIndexImpl<LocalCoords, const LocalCoords> IteratorImpl;

    /// Mesh iterator type.
    typedef PolymorphicForwardIteratorWithIndex<IteratorImpl> Iterator;

    // To be more compatibile with STL:
    typedef Iterator iterator;
    typedef const Iterator const_iterator;

    /// @return iterator at first point
    virtual Iterator begin() const = 0;

    /// @return iterator just after last point
    virtual Iterator end() const = 0;

    /**
     * Store information about event connected with geometry element.
     *
     * Subclasses of this can includes additional information about specific type of event.
     */
    struct Event: public EventWithSourceAndFlags< MeshD<dimension> > {

        /// Event flags (which describes event properties).
        enum Flags {
            DELETE = 1,             ///< is deleted
            RESIZE = 1<<1,          ///< size could be changed (points added or deleted)
            USER_DEFINED = 1<<2     ///< user-defined flags could have ids: USER_DEFINED, USER_DEFINED<<1, USER_DEFINED<<2, ...
        };

        /**
         * Check if given @p flag is set.
         * @param flag flag to check
         * @return @c true only if @p flag is set
         */
        bool hasFlag(Flags flag) const { return hasAnyFlag(flag); }

        /**
         * Check if DELETE flag is set, which mean that source of event is deleted.
         * @return @c true only if DELETE flag is set
         */
        bool isDelete() const { return hasFlag(DELETE); }

        /**
         * Check if RESIZE flag is set, which mean that source of event could be resized.
         * @return @c true only if RESIZE flag is set
         */
        bool isResize() const { return hasFlag(RESIZE); }

        /**
         * Construct event.
         * @param source source of event
         * @param flags flags which describes event's properties
         */
        explicit Event(MeshD<dimension>& source, unsigned char flags = 0):  EventWithSourceAndFlags< MeshD<dimension> >(source, flags) {}
    };

    /// Changed signal, fired when space was changed.
    boost::signals2::signal<void(Event&)> changed;

    template <typename ClassT, typename methodT>
    void changedConnectMethod(ClassT* obj, methodT method) {
        changed.connect(boost::bind(method, obj, _1));
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
        EventT evt(*this, std::forward<Args>(event_constructor_params_without_source)...);
        changed(evt);
    }

    void fireResized() { fireChanged(Event::RESIZE); }

    /**
     * Initialize this to be the same as @p to_copy but doesn't have any changes observer.
     * @param to_copy object to copy
     */
    MeshD(const MeshD& to_copy) {}

    /**
     * Set this to be the same as @p to_copy but doesn't changed changes observer.
     * @param to_copy object to copy
     */
    MeshD& operator=(const MeshD& to_copy) { return *this; }

    MeshD() = default;

    /// Inform observators that this is being deleted
    virtual ~MeshD() { fireChanged(Event::DELETE); }

};

/**
 * Implementation of Mesh::IteratorImpl.
 * Holds iterator of wrapped type (const_internal_iterator_t) and delegate all calls to it.
 */
template <typename const_internal_iterator_t, int dim = std::iterator_traits<const_internal_iterator_t>::value_type::DIMS>
struct MeshIteratorWrapperImpl: public MeshD<dim>::IteratorImpl {

    const_internal_iterator_t internal_iterator;

    MeshIteratorWrapperImpl(const const_internal_iterator_t& internal_iterator)
    : internal_iterator(internal_iterator) {}

    virtual const typename MeshD<dim>::LocalCoords dereference() const {
        return *internal_iterator;
    }

    virtual void increment() {
        ++internal_iterator;
    }

    virtual bool equal(const typename MeshD<dim>::IteratorImpl& other) const {
        return internal_iterator == static_cast<const MeshIteratorWrapperImpl<const_internal_iterator_t, dim>&>(other).internal_iterator;
    }

    virtual MeshIteratorWrapperImpl<const_internal_iterator_t, dim>* clone() const {
        return new MeshIteratorWrapperImpl<const_internal_iterator_t, dim>(internal_iterator);
    }

    virtual std::size_t getIndex() const {
        return internal_iterator.getIndex();
    }

};

/**
 * Construct MeshD<dim>::Iterator which wraps non-polymorphic iterator, using MeshIteratorWrapperImpl.
 * @param iter iterator to wrap
 * @return wrapper over @p iter
 * @tparam IteratorType type of iterator to wrap
 * @tparam dim number of dimensions of IteratorType and resulted iterator (can be auto-detected in most situations)
 */
template <typename IteratorType, int dim = std::iterator_traits<IteratorType>::value_type::DIMS>
inline typename MeshD<dim>::Iterator makeMeshIterator(IteratorType iter) {
    return typename MeshD<dim>::Iterator(new MeshIteratorWrapperImpl<IteratorType, dim>(iter));
}


/**
 *  Template which instantiation is a class inherited from plask::Mesh (it is a Mesh implementation).
 *
 *  It helds an @a internal mesh (of type InternalMeshType) and uses it to implement plask::Mesh methods.
 *  All constructors and -> calls are delegated to the @a internal mesh.
 *
 *  Example usage:
 *  @code
 *  // Create 3D mesh which uses std::vector of 3d points as internal representation:
 *  plask::SimpleMeshAdapter< std::vector< plask::Vec<3, double> >, 3 > mesh;
 *  // Append two points to vector:
 *  mesh.internal.push_back(plask::vec(1.0, 1.2, 3.0));
 *  mesh->push_back(plask::vec(3.0, 4.0, 0.0)); // mesh-> is a shortcut to mesh.internal.
 *  // Now, mesh contains two points:
 *  assert(mesh.size() == 2);
 *  @endcode
 *
 *  @tparam InternalMeshType Internal mesh type.
 *  It must:
 *  - allow for iterate (has begin() and end() methods) over Vec<dim, double>,
 *  - has size() method which return number of points in mesh.
 */
//TODO needs getIndex in iterators or another iterator wrapper which calculate this
template <typename InternalMeshType, int dim>
struct SimpleMeshAdapter: public MeshD<dim> {

    //typedef MeshIteratorWrapperImpl<typename InternalMeshType::const_iterator, dim> IteratorImpl;

    /// Held internal, usually optimized, mesh.
    InternalMeshType internal;

    /**
     * Delegate constructor to @a internal representation.
     * @param params parameters for InternalMeshType constructor
     */
    template<typename ...Args>
    SimpleMeshAdapter<InternalMeshType, dim>(Args&&... params)
    : internal(std::forward<Args>(params)...) {
    }

    /**
     * Delegate call to @a internal.
     * @return <code>&internal</code>
     */
    InternalMeshType* operator->() {
        return &internal;
    }

    /**
     * Delegate call to @a internal.
     * @return <code>&internal</code>
     */
    const InternalMeshType* operator->() const {
        return &internal;
    }

    // MeshD<dim> methods implementation:
    virtual std::size_t size() const { return internal.size(); }
    virtual typename MeshD<dim>::Iterator begin() const { return makeMeshIterator(internal.begin()); }
    virtual typename MeshD<dim>::Iterator end() const { return makeMeshIterator(internal.end()); }

};

/** Base template for rectangular mesh of any dimension */
template <int dim, typename Mesh1D>
class RectangularMesh {};


/** Base class for every mesh generator */
class MeshGenerator {
  public:
    virtual ~MeshGenerator() {}
};

/** Base class for specific mesh generator */
template <typename MeshT>
class MeshGeneratorOf: public MeshGenerator
{
  protected:
    Cache<GeometryElement, MeshT, CacheRemoveOnEachChange> cache;

  public:
    // Type of generated mesh
    typedef MeshT MeshType;

    /**
     * Generate new mesh
     * \param geometry on which the mesh should be generated
     * \return new generated mesh
     */
    virtual shared_ptr<MeshT> generate(const shared_ptr<GeometryElementD<MeshT::dim>>& geometry) = 0;

    /**
     * Clear the cache of generated meshes.
     * This method should be called each time any parameter of generator is changed
     */
    inline void clearCache() {
        cache.clear();
    }

    /// Get generated mesh if it is cached or create a new one
    shared_ptr<MeshT> operator()(const shared_ptr<GeometryElementD<MeshT::dim>>& geometry) {
        if (auto res = cache.get(geometry))
            return res;
        else
            return cache(geometry, generate(geometry));
    }

};

/**
 * Helper which call stores mesh reader when constructed.
 * Each mesh can create one global instance of this class to register its reader.
 */
struct RegisterMeshReader {
    typedef shared_ptr<Mesh> ReadingFunction(XMLReader&);
    RegisterMeshReader(const std::string& tag_name, ReadingFunction* fun);
    static std::map<std::string, ReadingFunction*>& getReaders();
    static ReadingFunction* getReader(const std::string& name);
};

class Manager;

/**
 * Helper which call stores mesh reader when constructed.
 * Each mesh can create one global instance of this class to register its reader.
 */
struct RegisterMeshGeneratorReader {
    typedef shared_ptr<MeshGenerator> ReadingFunction(XMLReader&, const Manager&);
    RegisterMeshGeneratorReader(const std::string& tag_name, ReadingFunction* fun);
    static std::map<std::string, ReadingFunction*>& getReaders();
    static ReadingFunction* getReader(const std::string& name);
};


} // namespace plask

#endif  //PLASK__MESH_H
