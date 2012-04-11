#ifndef PLASK__MESH_H
#define PLASK__MESH_H

/** @file
This file includes base classes for meshes.
@see @ref meshes
*/

/** @page meshes Meshes
@section meshes_about About meshes
The mesh represents (ordered) set of points in 2d or 3d space. All meshes in PLaSK implements (inherits from)
instantiation of plask::Mesh template interface.

Typically, there is some data associated with points in mesh.
In PLaSK, all this data is not stored in the mesh class, hence they must be stored separately.
As the points in the mesh are ordered and each one have unique index in a range from <code>0</code>
to <code>plask::Mesh::size()-1</code>,
you can store data in any indexed structure, like an array (1d) or std::vector (which is recommended),
storing the data value for the i-th point in the mesh under the i-th index.

@see @ref interpolation

@section meshes_write How to implement a new mesh?
There are two typical approaches to implementing new types of meshes:
- @ref meshes_write_adapters "using adapters" (this approach is recommended),
- @ref meshes_write_direct "direct".

@see @ref interpolation_write

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

@subsection meshes_write_direct Direct implementation of plask::Mesh\<DIM\>
To implement a new mesh directly you have to write class inherited from the plask::Mesh\<DIM\>, where DIM (is equal 2 or 3) is a number of dimension of space your mesh is defined over.

You are required to:
- implement the @ref plask::Mesh::size size method;
- implement the iterator over the mesh points, which required to:
  - writing class inherited from plask::Mesh::IteratorImpl (and implement all its abstract methods),
  - writing @ref plask::Mesh::begin "begin()" and @ref plask::Mesh::end "end()" methods, typically this methods only returns:
    @code plask::Mesh::Iterator(new YourIteratorImpl(...)) @endcode
  - see also: MeshIteratorWrapperImpl and makeMeshIterator

Example implementation of singleton mesh (mesh which represent set with only one point in 3d space):
@code
struct OnePoint3dMesh: public plask::Mesh<3> {

    //Held point:
    plask::Vec<3, double> point;

    OnePoint3dMesh(const plask::Vec<3, double>& point)
    : point(point) {}

    //Iterator:
    struct IteratorImpl: public Mesh<plask::space::Cartesian3d>::IteratorImpl {

        //point to mesh or is equal to nullptr for end iterator
        const OnePoint3dMesh* mesh_ptr;

        //mesh == nullptr for end iterator
        IteratorImpl(const OnePoint3dMesh* mesh)
        : mesh_ptr(mesh) {}

        virtual const plask::Vec<3, double> dereference() const {
            return mesh_ptr->point;
        }

        virtual void increment() {
            mesh_ptr = nullptr; //we iterate only over one point, so next state is end
        }

        virtual bool equal(const typename Mesh<plask::space::Cartesian3d>::IteratorImpl& other) const {
            return mesh_ptr == static_cast<const IteratorImpl&>(other).mesh_ptr;
        }

        virtual IteratorImpl* clone() const {
            return new IteratorImpl(mesh_ptr);
        }

    };

    //plask::Mesh<3> methods implementation:

    virtual std::size_t size() const {
        return 1;
    }

    virtual typename Mesh<plask::space::Cartesian3d>::Iterator begin() const {
        return Mesh<3>::Iterator(new IteratorImpl(this));
    }

    virtual typename Mesh<plask::space::Cartesian3d>::Iterator end() const {
        return Mesh<3>::Iterator(new IteratorImpl(nullptr));
    }

};
@endcode
You should also implement interpolation algorithms for your mesh, see @ref interpolation_write for more details.
*/

#include <plask/config.h>

#include "../vec.h"
#include "../utils/iterators.h"

namespace plask {

/**
 * Base class for all the meshes.
 * Mesh represent a set of points in 2d or 3d space and:
 * - knows number of points,
 * - allows for iterate over this points,
 * - can calculate interpolated value for given destination points, source values, and the interpolation method.
 *
 * @see @ref meshes
 */
template <int dimension>
struct Mesh {

    /// Number of dimensions
    static const int dim = dimension;

    /// @return number of points in mesh
    virtual std::size_t size() const = 0;

    ///@return true only if there are no points in mesh
    bool empty() const { return size() == 0; }

    /// Type of vector representing coordinates in local space
    typedef Vec<dim, double> LocalCoords;

    /// Base class for mesh iterator implementation.
    typedef PolymorphicForwardIteratorImpl<LocalCoords, const LocalCoords> IteratorImpl;

    /// Mesh iterator type.
    typedef PolymorphicForwardIterator<IteratorImpl> Iterator;

    // To be more compatibile with STL:
    typedef Iterator iterator;
    typedef const Iterator const_iterator;

    /// @return iterator at first point
    virtual Iterator begin() const = 0;

    /// @return iterator just after last point
    virtual Iterator end() const = 0;

};

/**
 * Implementation of Mesh::IteratorImpl.
 * Holds iterator of wrapped type (const_internal_iterator_t) and delegate all calls to it.
 */
template <typename const_internal_iterator_t, int dim = std::iterator_traits<const_internal_iterator_t>::value_type::DIMS>
struct MeshIteratorWrapperImpl: public Mesh<dim>::IteratorImpl {

    const_internal_iterator_t internal_iterator;

    MeshIteratorWrapperImpl(const const_internal_iterator_t& internal_iterator)
    : internal_iterator(internal_iterator) {}

    virtual const typename Mesh<dim>::LocalCoords dereference() const {
        return *internal_iterator;
    }

    virtual void increment() {
        ++internal_iterator;
    }

    virtual bool equal(const typename Mesh<dim>::IteratorImpl& other) const {
        return internal_iterator == static_cast<const MeshIteratorWrapperImpl<const_internal_iterator_t, dim>&>(other).internal_iterator;
    }

    virtual MeshIteratorWrapperImpl<const_internal_iterator_t, dim>* clone() const {
        return new MeshIteratorWrapperImpl<const_internal_iterator_t, dim>(internal_iterator);
    }

};

/**
 * Construct Mesh<dim>::Iterator which wraps non-polimorphic iterator, using MeshIteratorWrapperImpl.
 * @param iter iterator to wrap
 * @return wrapper over @p iter
 * @tparam IteratorType type of iterator to wrap
 * @tparam dim number of dimensions of IteratorType and resulted iterator (can be auto-detected in most situations)
 */
template <typename IteratorType, int dim = std::iterator_traits<IteratorType>::value_type::DIMS>
inline typename Mesh<dim>::Iterator makeMeshIterator(IteratorType iter) {
    return typename Mesh<dim>::Iterator(new MeshIteratorWrapperImpl<IteratorType, dim>(iter));
}


/**
Template which instantiation is a class inherited from plask::Mesh (it is a Mesh implementation).

It helds an @a internal mesh (of type InternalMeshType) and uses it to implement plask::Mesh methods.
All constructors and -> calls are delegated to the @a internal mesh.

Example usage:
@code
// Create 3D mesh which uses std::vector of 3d points as internal representation:
plask::SimpleMeshAdapter< std::vector< plask::Vec<3, double> >, 3 > mesh;
// Append two points to vector:
mesh.internal.push_back(plask::vec(1.0, 1.2, 3.0));
mesh->push_back(plask::vec(3.0, 4.0, 0.0)); // mesh-> is a shortcut to mesh.internal.
// Now, mesh contains two points:
assert(mesh.size() == 2);
@endcode

@tparam InternalMeshType Internal mesh type.
It must:
- allow for iterate (has begin() and end() methods) over Vec<dim, double>,
- has size() method which return number of points in mesh.
*/
template <typename InternalMeshType, int dim>
struct SimpleMeshAdapter: public Mesh<dim> {

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

    // Mesh<dim> methods implementation:
    virtual std::size_t size() const { return internal.size(); }
    virtual typename Mesh<dim>::Iterator begin() const { return makeMeshIterator(internal.begin()); }
    virtual typename Mesh<dim>::Iterator end() const { return makeMeshIterator(internal.end()); }

};

} // namespace plask

#endif  //PLASK__MESH_H
