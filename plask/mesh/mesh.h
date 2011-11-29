#ifndef PLASK__MESH_H
#define PLASK__MESH_H

/** @page meshes Meshes
@section meshes_about About
The mesh represents (ordered) set of points in 3d space. All meshes in PLaSK implements (inherits from)
plask::Mesh interface.

Typically, there is some data associated with points in mesh.
In PLaSK, all this data is not stored in the mesh class, hence they must be stored separately.
As the points in the mesh are ordered and each one have unique index in a range from <code>0</code>
to <code>plask::Mesh::size()-1</code>,
you can store data in any indexed structure, like an array or std::vector (which is recommended),
storing the data value for the i-th point in the mesh under the i-th index.

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

@section meshes_interpolation Data interpolation
PLaSK provides a mechanism to calculate (interpolate) a field of some physical quantity in arbitrary requested points,
if values of this field are known in different points.
Both sets of points are described by meshes and associated with the vectors of corresponding values.

Let us denote:
- @a src_mesh - set of points in which the field values are known
- @a src_vec - vector of known field values, in points described by the @a sec_mesh
- @a dst_mesh - requested set of points, in which field values should be calculated (interpolated)
- @a dst_vec - vector of field values in points described by the @a dst_mesh, to calculate

plask::interpolate method calculates and returns @a dst_vec for a given
@a src_mesh, @a src_vec, @a dst_mesh and interpolation method.

plask::interpolate can return a newly created vector or the @a src_vec if @a src_mesh and @a dst_mesh are the same.
Furthermore, the lifespan of both source and destination data cannot be determined in advance.
For this reason @a src_vec is passed and @a dst_vec is returned through an std::shared_ptr, a smart pointer
which is responsible for deleting the data in the proper time (i.e. when all the existing modules delete their copy
of the pointer, indicating they are not going to use this data any more). However, for this mechanism to work
efficiently, all the modules must allocate the data using the std::shared_ptr, as described in
@ref modules.

Typically, plask::interpolate is called inside providers of the fields of scalars or vectors (see @ref providers).
Note that the exact @a src_mesh type must be known at the compile time by the module providing the data
(usually this is the native mesh of the module). Interpolation algorithms depend on this type. On the other hand
the @a dst_mesh can be very generic and needs only to provide the iterator over its points. This mechanism allows
to connect the module providing some particular data with any module requesting it, regardless of its mesh.

@section meshes_write How to implement a new mesh?
There are two typical approaches to implementing new types of meshes:
- @ref meshes_write_direct "direct",
- @ref meshes_write_adapters "using adapters" (this approach is recommended).

@subsection meshes_write_direct Direct implementation of plask::Mesh
To implement a new mesh directly you have to write class inherited from the plask::Mesh. You are required to:
- implement the @ref plask::Mesh::size size method;
- implement the iterator over the mesh points, which required to:
  - writing class inherited from plask::Mesh::IteratorImpl (and implement all its abstract methods),
  - writing @ref plask::Mesh::begin "begin()" and @ref plask::Mesh::end "end()" methods (typically this methods only returns <code>plask::Mesh::Iterator(new YourIteratorImpl(...))</code>).

TODO: example

@subsection meshes_write_adapters Using adapters to generate plask::Mesh implementation
You can specialize adapter template to generate class which inherit from plask::Mesh.

To do this, you have to implement internal mesh representation class (see @ref meshes_internal) first.
Your class must fulfill adapter templates requirements (it is one of adapter template parameters),
and also can have extra methods for your internal use (for calculation).

Adapter templates currently available in PLaSK:
- plask::SimpleMeshAdapter

TODO: example (here or in adapters description)

@section interpolation_write How to write a new interpolation algorithm?

To implement an interpolation method (which you must usually do in any case where your mesh is the source mesh)
you have to write a specialization or a partial specialization of the plask::InterpolationAlgorithm class template
for specific: source mesh type, data type, and/or @ref plask::InterpolationMethod "interpolation method".
Your specialization must contain an implementation of the static method plask::InterpolationAlgorithm::interpolate.

For example to implement @ref plask::LINEAR "linear" interpolation for MyMeshType source mesh type using the same code for all possible data types, you write:
@code
template <typename DataT>    //for any data type
struct plask::InterpolationAlgorithm<MyMeshType, DataT, plask::LINEAR> {
    static void interpolate(MyMeshType& src_mesh, const std::vector<DataT>& src_vec, const plask::Mesh& dst_mesh, std::vector<DataT>& dst_vec)
    throw (plask::NotImplemented) {
        // here comes your interpolation code
    }
};
@endcode
Note that above code is template and must be placed in header file.

The next example, shows how to implement algorithm for a particular data type.
To implement the interpolation version for the 'double' type, you should write:
@code
template <>
struct plask::InterpolationAlgorithm<MyMeshType, double, plask::LINEAR> {
    static void interpolate(MyMeshType& src_mesh, const std::vector<double>& src_vec, const plask::Mesh& dst_mesh, std::vector<double>& dst_vec)
    throw (plask::NotImplemented) {
        // interpolation code for vectors of doubles
    }
};
@endcode
You can simultaneously have codes from both examples.
In such case, when linear interpolation from the source mesh MyMeshType is requested,
the compiler will use the second implementation to interpolate vectors of doubles
and the first one in all other cases.

The code of the function should iterate over all the points of the @a dst_mesh and fill the @a dst_vec with the interpolated values in the respective points.

TODO: write more explanations, and give some examples
*/

#include "../space.h"
#include <memory>

#include "../utils/iterators.h"
#include "interpolation.h"

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
struct Mesh {

    /// @return number of points in mesh
    virtual std::size_t size() const = 0;

    /**
     * Calculate (interpolate) a field of some physical properties in points described by this mesh
     * if values of this field in different points (@a src_mesh) are known.
     * @param src_mesh set of points in which the field values are known
     * @param src_vec vector of known field values in points described by @a sec_mesh
     * @param method interpolation method to use
     * @return vector of the field values in points described by this mesh,
     *         can be equal to @a src_vec if @a src_mesh and this mesh are the same mesh
     * @throw NotImplemented if given interpolation method is not implemented for used source mesh type
     * @throw CriticalException if given interpolation method is not valid
     */
    template <typename SrcMeshT, typename DataT>
    inline std::shared_ptr<const std::vector<DataT>>
    fill(SrcMeshT& src_mesh, std::shared_ptr<const std::vector<DataT>>& src_vec, InterpolationMethod method = DEFAULT)
         throw (NotImplemented, CriticalException) {
        return interpolate(src_mesh, src_vec, *this, method);
    }

    /// Base class for mesh iterator implementation.
    typedef PolymorphicForwardIteratorImpl< Vector3d<double>, const Vector3d<double> > IteratorImpl;

    /// Mesh iterator type.
    typedef PolymorphicForwardIterator< IteratorImpl > Iterator;

    // To be more compatibile with STL:
    typedef Iterator iterator;
    typedef Iterator const_iterator;

    /// @return iterator at first point
    virtual Iterator begin() = 0;

    /// @return iterate just after last point
    virtual Iterator end() = 0;

};

/**
Template which specialization is class inherited from Mesh (is Mesh implementation).
@tparam InternalMeshType Mesh type. Can be in diferent space.
It must:
    - InternalMeshType::PointType must be a typename of points used by InternalMeshType
    - allow for iterate (has begin and end methods) over InternalMeshType::PointType, and has defined InternalMeshType::const_iterator for constant iterator type
    - has size method
@tparam toPoint3d function which is used to convert points from InternalMeshType space to Vector3d<double> (used by plask::Mesh)
*/
template <typename InternalMeshType, Vector3d<double> (*toPoint3d)(typename InternalMeshType::PointType)>
struct SimpleMeshAdapter: public Mesh {

    /// Holded, internal, typically optimized mesh.
    InternalMeshType internal;

    /**
     * Implementation of Mesh::IteratorImpl.
     * Hold iterator of wrapped type (InternalMeshType::const_iterator) and delegate all calls to it.
     */
    struct IteratorImpl: public Mesh::IteratorImpl {

        typename InternalMeshType::const_iterator internal_iterator;

        IteratorImpl(typename InternalMeshType::const_iterator&& internal_iterator)
        : internal_iterator(std::forward(internal_iterator)) {}

        virtual Vector3d<double> dereference() const {
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


} // namespace plask

#endif  //PLASK__MESH_H
