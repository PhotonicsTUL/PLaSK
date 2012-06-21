#ifndef PLASK__INTERPOLATION_H
#define PLASK__INTERPOLATION_H

/** @page interpolation Interpolation
@section meshes_interpolation About interpolation
PLaSK provides a mechanism to calculate (interpolate) a field of some physical quantity in arbitrary requested points,
if values of this field are known in different points.
Both sets of points are described by @ref meshes "meshes" and associated with the vectors of corresponding values.

Let us denote:
- @a src_mesh - set of points in which the field values are known
- @a src_vec - vector of known field values, in points described by the @a sec_mesh
- @a dst_mesh - requested set of points, in which field values should be calculated (interpolated)
- @a dst_vec - vector of field values in points described by the @a dst_mesh, to calculate

plask::interpolate method calculates and returns @a dst_vec for a given
@a src_mesh, @a src_vec, @a dst_mesh and interpolation method.

plask::interpolate can return a newly created vector or the @a src_vec if @a src_mesh and @a dst_mesh are the same.
Furthermore, the lifespan of both source and destination data cannot be determined in advance.
For this reason @a src_vec is passed and @a dst_vec is returned through a DataVector class,
which is responsible for deleting the data in the proper time (i.e. when all the existing modules delete their copies
of the pointer, indicating they are not going to use this data any more). However, for this mechanism to work
efficiently, all the modules should allocate the data using DataVector, as described in
@ref modules.

Typically, plask::interpolate is called inside providers of the fields of scalars or vectors (see @ref providers).
Note that the exact @a src_mesh type must be known at the compile time by the module providing the data
(usually this is the native mesh of the module). Interpolation algorithms depend on this type. On the other hand
the @a dst_mesh can be very generic and needs only to provide the iterator over its points. This mechanism allows
to connect the module providing some particular data with any module requesting it, regardless of its mesh.


@section interpolation_write How to write a new interpolation algorithm?

To implement an interpolation method (which you must usually do in any case where your mesh is the source mesh)
you have to write a specialization or a partial specialization of the plask::InterpolationAlgorithm class template
for specific: source mesh type, data type, and/or @ref plask::InterpolationMethod "interpolation method".
Your specialization must contain an implementation of the static method plask::InterpolationAlgorithm::interpolate.

For example to implement @ref plask::LINEAR "linear" interpolation for MyMeshType source mesh type using the same code for all possible data types, you write:
@code
template <typename DataT>    //for any data type
struct plask::InterpolationAlgorithm<MyMeshType, DataT, plask::LINEAR> {
    static void interpolate(MyMeshType& src_mesh, const DataVector<DataT>& src_vec, const plask::Mesh<MyMeshType::dim>& dst_mesh, DataVector<DataT>& dst_vec) {

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
    static void interpolate(MyMeshType& src_mesh, const DataVector<double>& src_vec, const plask::Mesh<MyMeshType::dim>& dst_mesh, DataVector<double>& dst_vec) {

        // interpolation code for vectors of doubles
    }
};
@endcode
You can simultaneously have codes from both examples.
In such case, when linear interpolation from the source mesh MyMeshType is requested,
the compiler will use the second implementation to interpolate vectors of doubles
and the first one in all other cases.

Typically, the code of the function should iterate over all the points of the @a dst_mesh and fill the @a dst_vec with the interpolated values in the respective points.
*/

#include <typeinfo>  // for 'typeid'

#include "mesh.h"
#include "../exceptions.h"
#include "../memory.h"
#include "../data.h"

namespace plask {

/**
 * Supported interpolation methods.
 * @see @ref meshes_interpolation
 */
enum InterpolationMethod {
    DEFAULT_INTERPOLATION = 0,  ///< default interpolation (depends on source mesh)
    LINEAR = 1,                 ///< linear interpolation
    SPLINE = 2,                 ///< spline interpolation
    COSINE = 3,                 ///< cosine interpolation
    FOURIER = 4,                ///< Fourier transform interpolation
    //...add new interpolation algorithms here...
#   ifndef DOXYGEN
    __ILLEGAL_INTERPOLATION_METHOD__  // necessary for metaprogram loop and automatic Python enums
#   endif // DOXYGEN
};

static const char* interpolationMethodNames[] = { "DEFAULT", "LINEAR", "SPLINE", "COSINE", "FOURIER" /*attach new interpolation algoritm names here*/};

/**
 * Specialization of this class are used for interpolation and can depend on source mesh type,
 * data type and the interpolation method.
 * @see @ref interpolation_write
 */
template <typename SrcMeshT, typename DataT, InterpolationMethod method>
struct InterpolationAlgorithm
{
    static void interpolate(SrcMeshT& src_mesh, const DataVector<DataT>& src_vec, const Mesh<SrcMeshT::dim>& dst_mesh, DataVector<DataT>& dst_vec) {
        std::string msg = "interpolate (source mesh type: ";
        msg += typeid(src_mesh).name();
        msg += ", interpolation method: ";
        msg += interpolationMethodNames[method];
        msg += ")";
        throw NotImplemented(msg);
        //TODO iterate over dst_mesh and call InterpolationAlgorithmForPoint
    }
};

#ifndef DOXYGEN
// The following structures are solely used for metaprogramming
template <typename SrcMeshT, typename DataT, int iter>
struct __InterpolateMeta__
{
    inline static void interpolate(SrcMeshT& src_mesh, const DataVector<DataT>& src_vec,
                Mesh<SrcMeshT::dim>& dst_mesh, DataVector<DataT>& dst_vec, InterpolationMethod method) {
        if (int(method) == iter)
            InterpolationAlgorithm<SrcMeshT, DataT, (InterpolationMethod)iter>::interpolate(src_mesh, src_vec, dst_mesh, dst_vec);
        else
            __InterpolateMeta__<SrcMeshT, DataT, iter+1>::interpolate(src_mesh, src_vec, dst_mesh, dst_vec, method);
    }
};
template <typename SrcMeshT, typename DataT>
struct __InterpolateMeta__<SrcMeshT, DataT, __ILLEGAL_INTERPOLATION_METHOD__>
{
    inline static void interpolate(SrcMeshT& src_mesh, const DataVector<DataT>& src_vec,
                Mesh<SrcMeshT::dim>& dst_mesh, DataVector<DataT>& dst_vec, InterpolationMethod method) {
        throw CriticalException("no such interpolation method");
    }
};
#endif // DOXYGEN


/**
 * Calculate (interpolate when needed) a field of some physical properties in requested points of (@a dst_mesh)
 * if values of this field in points of (@a src_mesh) are known.
 * @param src_mesh set of points in which fields values are known
 * @param src_vec_ptr pointer to the vector of known field values in points described by @a sec_mesh
 * @param dst_mesh requested set of points, in which the field values should be calculated (interpolated)
 * @param method interpolation method to use
 * @return vector of the field values in points described by @a dst_mesh, can be equal to @a src_vec
 *         if @a src_mesh and @a dst_mesh are the same mesh
 * @throw NotImplemented if given interpolation method is not implemented for used source mesh type
 * @throw CriticalException if given interpolation method is not valid
 * @see @ref meshes_interpolation
 */
template <typename SrcMeshT, typename DataT>
inline const DataVector<DataT>
interpolate(SrcMeshT& src_mesh, const DataVector<DataT> src_vec_ptr,
            Mesh<SrcMeshT::dim>& dst_mesh, InterpolationMethod method = DEFAULT_INTERPOLATION)
{
    if (&src_mesh == &dst_mesh) return src_vec_ptr; // meshes are identical, so just return src_vec

    DataVector<DataT> result(dst_mesh.size());
    __InterpolateMeta__<SrcMeshT, DataT, 0>::interpolate(src_mesh, src_vec_ptr, dst_mesh, result, method);
    return result;
}

#ifndef DOXYGEN
// This is necessary for passing non-const src_vec_ptr.
// Apparently C++ has problems with proper casting is the vector is template argument of shared_ptr
/*template <typename SrcMeshT, typename DataT>
inline shared_ptr<const std::vector<DataT>>
interpolate(SrcMeshT& src_mesh, shared_ptr<std::vector<DataT>> src_vec_ptr,
            Mesh<SrcMeshT::dim>& dst_mesh, InterpolationMethod method = DEFAULT_INTERPOLATION) {
    return interpolate(src_mesh, (shared_ptr<const std::vector<DataT>>&&)src_vec_ptr, dst_mesh, method);
}*/
#endif // DOXYGEN

} // namespace plask

#endif  //PLASK__INTERPOLATION_H
