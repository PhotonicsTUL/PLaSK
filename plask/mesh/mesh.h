#ifndef PLASK__MESH_H
#define PLASK__MESH_H

/** @file
This file includes base classes for meshes.
@see @ref meshes
*/

/** @page meshes Meshes
@section meshes_about About meshes
The mesh represents (ordered) set of points in 3d space. All meshes in PLaSK implements (inherits from)
plask::Mesh interface.

Typically, there is some data associated with points in mesh.
In PLaSK, all this data is not stored in the mesh class, hence they must be stored separately.
As the points in the mesh are ordered and each one have unique index in a range from <code>0</code>
to <code>plask::Mesh::size()-1</code>,
you can store data in any indexed structure, like an array or std::vector (which is recommended),
storing the data value for the i-th point in the mesh under the i-th index.

@see @ref interpolation

@section meshes_internal Internal representation
Most of mesh classes in PLaSK have @c internal field which is internal mesh interface.
Typically this interface:
- represents set of point in space different than 3d (for example 1d or 2d);
- allow for faster calculation than generic mesh interface, and often has more futures (methods);
- can be different for different types of meshes (there are no common base class for internal interfaces).

In most cases, mesh use @c internal field methods to implement itself methods (especially, abstract methods of plask::Mesh),
which required translation of points between 3d space (required by plask::Mesh interface)
and space used by internal representation.
Many mesh types can use objects with the same types as internal representation,
but realize different translation strategy.

@section meshes_write How to implement a new mesh?
There are two typical approaches to implementing new types of meshes:
- @ref meshes_write_direct "direct",
- @ref meshes_write_adapters "using adapters" (this approach is recommended).

@see @ref interpolation_write

@subsection meshes_write_direct Direct implementation of plask::Mesh\<SPACE\>
To implement a new mesh directly you have to write class inherited from the plask::Mesh\<SPACE\>, where space is a type of space your mesh is defined over.
It can be one of: @a SpaceXY (2D Cartesian coordinates), @a SpaceRZ (2D cylindrical coordinates) or @a SpaceXYZ (3D cooridinates).
You are required to:
- implement the @ref plask::Mesh::size size method;
- implement the iterator over the mesh points, which required to:
  - writing class inherited from plask::Mesh::IteratorImpl (and implement all its abstract methods),
  - writing @ref plask::Mesh::begin "begin()" and @ref plask::Mesh::end "end()" methods (typically this methods only returns <code>plask::Mesh::Iterator(new YourIteratorImpl(...))</code>).

TODO: example

@subsection spaces_and_coordinates A note about spaces and coordinates

TODO

@subsection meshes_write_adapters Using adapters to generate plask::Mesh implementation
You can specialize adapter template to generate class which inheritting from plask::Mesh. TODO

To do this, you have to implement internal mesh representation class (see @ref meshes_internal) first.
Your class must fulfill adapter templates requirements (it is one of adapter template parameters),
and also can have extra methods for your internal use (for calculation).

Adapter templates currently available in PLaSK:
- plask::SimpleMeshAdapter

TODO: example (here or in adapters description)
*/

#include "../space.h"
#include <boost/shared_ptr.hpp>

#include "../utils/iterators.h"

namespace plask {

/**
 * Base class for all the meshes.
 * Mesh represent a set of points in 3d space and:
 * - knows number of points,
 * - allows for iterate over this points,
 * - can calculate interpolated value for given destination points (in 3d), source values, and the interpolation method.
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
Template which specialization is class inherited from Mesh (is Mesh implementation).
@tparam InternalMeshType Mesh type.
It must:
    - InternalMeshType::PointType must be a typename of points used by InternalMeshType
    - allow for iterate (has begin and end methods) over InternalMeshType::PointType, and has defined InternalMeshType::const_iterator for constant iterator type
    - has size method
@tparam toPoint3d function which is used to convert points from InternalMeshType space to Vec3<double> (used by plask::Mesh)
* /
template <typename InternalMeshType, Vec3<double> (*toPoint3d)(typename InternalMeshType::PointType)>
struct SimpleMeshAdapter: public Mesh {

    /// Holded, internal, typically optimized mesh.
    InternalMeshType internal;

    / **
     * Implementation of Mesh::IteratorImpl.
     * Holds iterator of wrapped type (InternalMeshType::const_iterator) and delegate all calls to it.
     * /
    struct IteratorImpl: public Mesh::IteratorImpl {

        typename InternalMeshType::const_iterator internal_iterator;

        IteratorImpl(typename InternalMeshType::const_iterator&& internal_iterator)
        : internal_iterator(std::forward(internal_iterator)) {}

        virtual Vec3<double> dereference() const {
            return toPoint3d(*internal_iterator);
        }

        virtual void increment() {
            ++internal_iterator;
        }

        virtual bool equal(const Mesh::IteratorImpl& other) const {
            return internal_iterator == static_cast<IteratorImpl&>(other).internal_iterator;
        }

    };

    virtual std::size_t size() const {
        return internal.size();
    }

    virtual Mesh::Iterator begin() { return Mesh::Iterator(internal.begin()); }

    virtual Mesh::Iterator end() { return Mesh::Iterator(internal.end()); }

};
*/

} // namespace plask

#endif  //PLASK__MESH_H
