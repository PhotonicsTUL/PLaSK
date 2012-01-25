#ifndef PLASK__MESH_H
#define PLASK__MESH_H

/** @file
This file includes base classes for meshes.
@see @ref meshes
*/

/** @page meshes Meshes
@section meshes_about About meshes
The mesh represents (ordered) set of points in 2d or 3d space. All meshes in PLaSK implements (inherits from)
specialization of plask::Mesh template interface.

Typically, there is some data associated with points in mesh.
In PLaSK, all this data is not stored in the mesh class, hence they must be stored separately.
As the points in the mesh are ordered and each one have unique index in a range from <code>0</code>
to <code>plask::Mesh::size()-1</code>,
you can store data in any indexed structure, like an array (1d) or std::vector (which is recommended),
storing the data value for the i-th point in the mesh under the i-th index.

@see @ref interpolation

@section meshes_internal Internal representation
Most of mesh classes in PLaSK have @c internal field which is internal mesh interface.
Typically this interface:
- represents set of point in the same space as parent mesh;
- allow for faster calculation than generic mesh interface, and often has more futures (methods);
- can have different types (there are no common base class for internal interfaces).

In most cases, mesh use @c internal field methods to implement itself methods (especially, abstract methods of plask::Mesh).

@section meshes_write How to implement a new mesh?
There are two typical approaches to implementing new types of meshes:
- @ref meshes_write_adapters "using adapters" (this approach is recommended),
- @ref meshes_write_direct "direct".

@see @ref interpolation_write

@subsection meshes_write_adapters Using adapters to generate plask::Mesh implementation
You can specialize adapter template to generate class which inheriting from plask::Mesh specialization.

To do this, you have to implement internal mesh representation class (see @ref meshes_internal) first.
Your class must fulfill adapter templates requirements (it is one of adapter template parameters),
and also can have extra methods for your internal use (for calculation).

Adapter templates currently available in PLaSK (see its description for more details and examples):
- plask::SimpleMeshAdapter

@subsection meshes_write_direct Direct implementation of plask::Mesh\<SPACE\>
To implement a new mesh directly you have to write class inherited from the plask::Mesh\<SPACE\>, where space is a type of space your mesh is defined over.
It can be one of (see @a spaces_and_coordinates for more details):
- plask::space::Cartesian2d - 2D Cartesian coordinates,
- plask::space::Cylindrical2d - 2D cylindrical coordinates,
- plask::space::Cartesian3d - 3D (Cartesian) coordinates.
You are required to:
- implement the @ref plask::Mesh::size size method;
- implement the iterator over the mesh points, which required to:
  - writing class inherited from plask::Mesh::IteratorImpl (and implement all its abstract methods),
  - writing @ref plask::Mesh::begin "begin()" and @ref plask::Mesh::end "end()" methods (typically this methods only returns <code>plask::Mesh::Iterator(new YourIteratorImpl(...))</code>).

Example implementation of singleton mesh (mesh which represent set with only one point in 3d space):
@code
struct OnePoint3dMesh: public plask::Mesh<plask::space::Cartesian3d> {
    
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
            return mesh_ptr == static_cast<IteratorImpl&>(other).mesh_ptr;
        }
        
        virtual IteratorImpl* clone() const {
            return new IteratorImpl(mesh_ptr);
        }

    }; 
    
    //plask::Mesh<plask::space::Cartesian3d> methods implementation:
    
    virtual std::size_t size() const {
        return 1;
    }

    virtual typename Mesh<plask::space::Cartesian3d>::Iterator begin() {
        return Mesh<plask::space::Cartesian3d>::Iterator(new IteratorImpl(this));
    }

    virtual typename Mesh<plask::space::Cartesian3d>::Iterator end() {
        return Mesh<plask::space::Cartesian3d>::Iterator(new IteratorImpl(nullptr));
    }
    
};
@endcode
You should also implement interpolation algorithms for your mesh, see @ref interpolation_write for more details.

@subsection spaces_and_coordinates A note about spaces and coordinates

Types: plask::space::Cartesian2d (2D Cartesian coordinates), plask::space::Cylindrical2d (2D cylindrical coordinates) and plask::space::Cartesian3d (3D cooridinates) (all defined in space.h)
described spaces. Each of this type includes static const and typedefs connected with space. You shouldn't construct object of this types.
*/

#include "../space.h"
#include <plask/config.h>

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
template <typename S>
struct Mesh {

    /// @return number of points in mesh
    virtual std::size_t size() const = 0;

    /// Type of the space over which the mesh is defined
    typedef S Space;

    /// Type of vector representing coordinates in local space
    typedef typename S::CoordsType LocalCoords;

    /// Base class for mesh iterator implementation.
    typedef PolymorphicForwardIteratorImpl<LocalCoords, const LocalCoords> IteratorImpl;

    /// Mesh iterator type.
    typedef PolymorphicForwardIterator<IteratorImpl> Iterator;

    // To be more compatibile with STL:
    typedef Iterator iterator;
    typedef const Iterator const_iterator;

    /// @return iterator at first point
    virtual Iterator begin() = 0;

    /// @return iterate just after last point
    virtual Iterator end() = 0;

};


/**
Template which specialization is class inherited from plask::Mesh (is Mesh implementation).

It held @a internal mesh (of type InternalMeshType) and use it to implement plask::Mesh methods.
All constructor and -> calls are delegated to the @a internal mesh.

Example usage:
@code
//Create 3d mesh which use std::vector of 3d points as internal representation:
plask::SimpleMeshAdapter< std::vector< plask::Vec<3, double> >, plask::space::Cartesian3d > mesh;
//Append two point to vector:
mesh.internal.push_back(plask::vec(1.0, 1.2, 3.0));
mesh->push_back(plask::vec(3.0, 4.0, 0.0)); //mesh-> is shortcut to mesh.internal.
//Now, mesh consist of two points:
assert(mesh.size() == 2);
@endcode

@tparam InternalMeshType Internal mesh type.
It must:
- allow for iterate (has begin() and end() methods) over Space::CoordsType, and has defined InternalMeshType::const_iterator for constant iterator type,
- has size() method which return number of points in mesh.
*/
template <typename InternalMeshType, typename Space>
struct SimpleMeshAdapter: public Mesh<Space> {

    /// Held, internal, typically optimized mesh.
    InternalMeshType internal;
    
    /**
     * Delegate constructor to @a internal representation.
     * @param params parameters for InternalMeshType constructor
     */
    template<typename ...Args>
    SimpleMeshAdapter<InternalMeshType, Space>(Args&&... params)
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

    /**
     * Implementation of Mesh::IteratorImpl.
     * Holds iterator of wrapped type (InternalMeshType::const_iterator) and delegate all calls to it.
     */
    struct IteratorImpl: public Mesh<Space>::IteratorImpl {

        typename InternalMeshType::const_iterator internal_iterator;

        IteratorImpl(const typename InternalMeshType::const_iterator& internal_iterator)
        : internal_iterator(internal_iterator) {}

        virtual const typename Mesh<Space>::LocalCoords dereference() const {
            return *internal_iterator;
        }

        virtual void increment() {
            ++internal_iterator;
        }

        virtual bool equal(const typename Mesh<Space>::IteratorImpl& other) const {
            return internal_iterator == static_cast<const IteratorImpl&>(other).internal_iterator;
        }
        
        virtual IteratorImpl* clone() const {
            return new IteratorImpl(internal_iterator);
        }

    };

    virtual std::size_t size() const {
        return internal.size();
    }

    virtual typename Mesh<Space>::Iterator begin() { return typename Mesh<Space>::Iterator(new IteratorImpl(internal.begin())); }

    virtual typename Mesh<Space>::Iterator end() { return typename Mesh<Space>::Iterator(new IteratorImpl(internal.end())); }

};

} // namespace plask

#endif  //PLASK__MESH_H
