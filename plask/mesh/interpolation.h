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
which is responsible for deleting the data in the proper time (i.e. when all the existing solvers delete their copies
of the pointer, indicating they are not going to use this data any more). However, for this mechanism to work
efficiently, all the solvers should allocate the data using DataVector, as described in
@ref solvers.

Typically, plask::interpolate is called inside providers of the fields of scalars or vectors (see @ref providers).
Note that the exact @a src_mesh type must be known at the compile time by the solver providing the data
(usually this is the native mesh of the solver). Interpolation algorithms depend on this type. On the other hand
the @a dst_mesh can be very generic and needs only to provide the iterator over its points. This mechanism allows
to connect the solver providing some particular data with any solver requesting it, regardless of its mesh.


@section interpolation_write How to write a new interpolation algorithm?

To implement an interpolation method (which you must usually do in any case where your mesh is the source mesh)
you have to write a specialization or a partial specialization of the plask::InterpolationAlgorithm class template
for specific: source mesh type, data type, and/or @ref plask::InterpolationMethod "interpolation method".
Your specialization must contain an implementation of the static method plask::InterpolationAlgorithm::interpolate.

For example to implement @ref plask::INTERPOLATION_LINEAR "linear" interpolation for MyMeshType source mesh type using the same code for all possible data types, you write:
@code
template <typename SrcT, typename DstT>    // for any data type
struct plask::InterpolationAlgorithm<MyMeshType, SrcT, DstT, plask::INTERPOLATION_LINEAR> {
    static void interpolate(MyMeshType& src_mesh, const DataVector<const SrcT>& src_vec, const plask::MeshD<MyMeshType::DIM>& dst_mesh, DataVector<DstT>& dst_vec) {

        // here comes your interpolation code
    }
};
@endcode
Note that above code is template and must be placed in header file.

The next example, shows how to implement algorithm for a particular data type.
To implement the interpolation version for the 'double' type, you should write:
@code
template <>
struct plask::InterpolationAlgorithm<MyMeshType, double, plask::INTERPOLATION_LINEAR> {
    static void interpolate(MyMeshType& src_mesh, const DataVector<const double>& src_vec, const plask::MeshD<MyMeshType::DIM>& dst_mesh, DataVector<double>& dst_vec) {

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
#include "../lazy_data.h"
#include <plask/log/log.h>

namespace plask {

/**
 * Supported interpolation methods.
 * @see @ref meshes_interpolation
 */
enum InterpolationMethod: unsigned {
    INTERPOLATION_DEFAULT = 0,  ///< Default interpolation (depends on source mesh)
    INTERPOLATION_NEAREST,      ///< Nearest neighbor interpolation
    INTERPOLATION_LINEAR,       ///< Linear interpolation
    INTERPOLATION_SPLINE,       ///< Spline interpolation with parabolic derivatives and Hyman monotonic filter
    INTERPOLATION_SMOOTH_SPLINE,///< Spline interpolation with continous second derivatives
    INTERPOLATION_PERIODIC_SPLINE,///< Spline interpolation with continous second derivatives and periodic edges (1D only)
    INTERPOLATION_FOURIER,      ///< Fourier transform interpolation
    // ...add new interpolation algorithms here...
#   ifndef DOXYGEN
    __ILLEGAL_INTERPOLATION_METHOD__  // necessary for metaprogram loop and automatic Python enums
#   endif // DOXYGEN
};

static constexpr const char* interpolationMethodNames[] = {
    "DEFAULT",
    "NEAREST",
    "LINEAR",
    "SPLINE",
    "SMOOTH_SPLINE",
    "PERIODIC_SPLINE",
    "FOURIER",
    // ...attach new interpolation algorithm names here...
    "ILLEGAL"
};

/**
 * Helper utility that replaces INTERPOLATION_DEFAULT with particular method.
 */
template <InterpolationMethod default_method>
inline InterpolationMethod defInterpolation(InterpolationMethod method) {
    return (method == INTERPOLATION_DEFAULT)? default_method : method;
}

/**
 * Specialization of this class are used for interpolation and can depend on source mesh type,
 * data type and the interpolation method.
 * @see @ref interpolation_write
 */
template <typename SrcMeshT, typename SrcT, typename DstT, InterpolationMethod method>
struct InterpolationAlgorithm
{
    static LazyData<DstT> interpolate(const shared_ptr<const SrcMeshT>& src_mesh, const DataVector<const SrcT>&, const shared_ptr<const MeshD<SrcMeshT::DIM>>&) {
        std::string msg = "interpolate (source mesh type: ";
        msg += typeid(*src_mesh).name();
        msg += ", interpolation method: ";
        msg += interpolationMethodNames[method];
        msg += ")";
        throw NotImplemented(msg);
    }
};

/**
 * Specialization of InterpolationAlgorithm showing elegant message if algorithm default is used.
 */
template <typename SrcMeshT, typename SrcT, typename DstT>
struct InterpolationAlgorithm<SrcMeshT, SrcT, DstT, INTERPOLATION_DEFAULT>
{
    static LazyData<DstT> interpolate(const shared_ptr<const SrcMeshT>&, const DataVector<const SrcT>&, const shared_ptr<const MeshD<SrcMeshT::DIM>>&) {
        throw CriticalException("interpolate(...) called for INTERPOLATION_DEFAULT method. Contact solver author to fix this issue."
#ifndef NDEBUG
                                "\n\nINFO FOR SOLVER AUTHOR: To avoid this error use 'defInterpolation<YOUR_DEFAULT_METHOD>(interpolation_method) in C++ code of the provider in your solver.\n"
#endif
                               );
    }
};

#ifndef DOXYGEN
// The following structures are solely used for metaprogramming
template <typename SrcMeshT, typename SrcT, typename DstT, int iter>
struct __InterpolateMeta__
{
    inline static LazyData<typename std::remove_const<DstT>::type> interpolate(const shared_ptr<const SrcMeshT>& src_mesh, const DataVector<const SrcT>& src_vec,
                                   const shared_ptr<const MeshD<SrcMeshT::DIM>>& dst_mesh, InterpolationMethod method) {
        if (int(method) == iter)
            return InterpolationAlgorithm<SrcMeshT, SrcT, typename std::remove_const<DstT>::type, (InterpolationMethod)iter>::interpolate(src_mesh, DataVector<const SrcT>(src_vec), dst_mesh);
        else
            return __InterpolateMeta__<SrcMeshT, SrcT, DstT, iter+1>::interpolate(src_mesh, src_vec, dst_mesh, method);
    }
};
template <typename SrcMeshT, typename SrcT, typename DstT>
struct __InterpolateMeta__<SrcMeshT, SrcT, DstT, __ILLEGAL_INTERPOLATION_METHOD__>
{
    inline static LazyData<typename std::remove_const<DstT>::type> interpolate(const shared_ptr<const SrcMeshT>&, const DataVector<const SrcT>&,
                                   const shared_ptr<const MeshD<SrcMeshT::DIM>>&, InterpolationMethod) {
        throw CriticalException("no such interpolation method");
    }
};
#endif // DOXYGEN


/**
 * Calculate (interpolate when needed) a field of some physical properties in requested points of (@a dst_mesh)
 * if values of this field in points of (@a src_mesh) are known.
 * @param src_mesh set of points in which fields values are known
 * @param src_vec the vector of known field values in points described by @a sec_mesh
 * @param dst_mesh requested set of points, in which the field values should be calculated (interpolated)
 * @param method interpolation method to use
 * @param verbose if true, the log message is written
 * @return vector of the field values in points described by @a dst_mesh, can be equal to @a src_vec
 *         if @a src_mesh and @a dst_mesh are the same mesh
 * @throw NotImplemented if given interpolation method is not implemented for used source mesh type
 * @throw CriticalException if given interpolation method is not valid
 * @see @ref meshes_interpolation
 */
template <typename SrcMeshT, typename SrcT, typename DstT=SrcT>
LazyData<typename std::remove_const<DstT>::type> interpolate(
                                        shared_ptr<const SrcMeshT> src_mesh,
                                        DataVector<const SrcT> src_vec,
                                        shared_ptr<const MeshD<SrcMeshT::DIM>> dst_mesh,
                                        InterpolationMethod method=INTERPOLATION_DEFAULT,
                                        bool verbose=true)
{
    if (src_mesh->size() != src_vec.size())
        throw BadMesh("interpolate", "Mesh size (%2%) and values size (%1%) do not match", src_vec.size(), src_mesh->size());
    if (src_mesh == dst_mesh) return new LazyDataFromVectorImpl<typename std::remove_const<DstT>::type>(src_vec); // meshes are identical, so just return src_vec
    if (verbose) writelog(LOG_DETAIL, std::string("interpolate: Running ") + interpolationMethodNames[method] + " interpolation");
    return __InterpolateMeta__<SrcMeshT, SrcT, DstT, 0>::interpolate(src_mesh, src_vec, dst_mesh, method);
}

template <typename SrcMeshT, typename SrcT, typename DstT=SrcT, typename DstMeshT>
LazyData<typename std::remove_const<DstT>::type> interpolate(
                            shared_ptr<SrcMeshT> src_mesh,
                            DataVector<SrcT> src_vec,
                            shared_ptr<DstMeshT> dst_mesh,
                            InterpolationMethod method=INTERPOLATION_DEFAULT,
                            bool verbose=true)
{
    return interpolate(shared_ptr<const SrcMeshT>(src_mesh), DataVector<const SrcT>(src_vec), shared_ptr<const MeshD<SrcMeshT::DIM>>(dst_mesh), method, verbose);
}

/*template <typename SrcMeshT, typename SrcT, typename DstT=SrcT>
LazyData<typename std::remove_const<DstT>::type> interpolate(shared_ptr<SrcMeshT> src_mesh, DataVector<const SrcT> src_vec,
                             shared_ptr<MeshD<SrcMeshT::DIM>> dst_mesh,
                             InterpolationMethod method=INTERPOLATION_DEFAULT, bool verbose=true)
{
    return interpolate(shared_ptr<const SrcMeshT>(src_mesh), src_vec, shared_ptr<const MeshD<SrcMeshT::DIM>>(dst_mesh), method, verbose);
}

template <typename SrcMeshT, typename SrcT, typename DstT=SrcT>
LazyData<typename std::remove_const<DstT>::type> interpolate(shared_ptr<const SrcMeshT> src_mesh, DataVector<const SrcT> src_vec,
                             shared_ptr<MeshD<SrcMeshT::DIM>> dst_mesh,
                             InterpolationMethod method=INTERPOLATION_DEFAULT, bool verbose=true)
{
    return interpolate(src_mesh, src_vec, shared_ptr<const MeshD<SrcMeshT::DIM>>(dst_mesh), method, verbose);
}*/

/**
 * Base class for lazy data (vector) that perform interpolation.
 *
 * It has reference to source (src_mesh) and destination (dst_mesh) meshes and to source data vector (src_vec).
 */
template <typename T, typename SrcMeshType>
struct InterpolatedLazyDataImpl: public LazyDataImpl<T> {

    shared_ptr<const SrcMeshType> src_mesh;
    shared_ptr<const MeshD<SrcMeshType::DIM>> dst_mesh;
    DataVector<const T> src_vec;

    InterpolatedLazyDataImpl(const shared_ptr<const SrcMeshType>& src_mesh, const DataVector<const T>& src_vec, const shared_ptr<const MeshD<SrcMeshType::DIM>>& dst_mesh)
        : src_mesh(src_mesh), dst_mesh(dst_mesh), src_vec(src_vec) {}

    virtual std::size_t size() const override { return dst_mesh->size(); }

};

/**
 * Implementation of InterpolatedLazyDataImpl which call src_mesh.interpolateLinear(src_vec, dst_mesh).
 *
 * So it can be used if SrcMeshType has such interpolateLinear method.
 */
template <typename T, typename SrcMeshType>
struct LinearInterpolatedLazyDataImpl: public InterpolatedLazyDataImpl<T, SrcMeshType> {

    LinearInterpolatedLazyDataImpl(shared_ptr<const SrcMeshType> src_mesh, const DataVector<const T>& src_vec, shared_ptr<const MeshD<SrcMeshType::DIM>> dst_mesh):
        InterpolatedLazyDataImpl<T, SrcMeshType>(src_mesh, src_vec, dst_mesh) {}

    virtual T get(std::size_t index) const override {
        return this->src_mesh->interpolateLinear(this->src_vec, this->dst_mesh->at(index));
    }

};

/**
 * Implementation of InterpolatedLazyDataImpl which call src_mesh.interpolateNearestNeighbor(src_vec, dst_mesh).
 *
 * So it can be used if SrcMeshType has such interpolateLinear method.
 */
template <typename T, typename SrcMeshType>
struct NearestNeighborInterpolatedLazyDataImpl: public InterpolatedLazyDataImpl<T, SrcMeshType> {

    NearestNeighborInterpolatedLazyDataImpl(shared_ptr<const SrcMeshType> src_mesh, const DataVector<const T>& src_vec, shared_ptr<const MeshD<SrcMeshType::DIM>> dst_mesh):
        InterpolatedLazyDataImpl<T, SrcMeshType>(src_mesh, src_vec, dst_mesh) {}

    virtual T get(std::size_t index) const override {
        return this->src_mesh->interpolateNearestNeighbor(this->src_vec, this->dst_mesh->at(index));
    }

};



} // namespace plask


namespace boost {

    template <>
    inline plask::InterpolationMethod lexical_cast<plask::InterpolationMethod, std::string>(const std::string& arg) {
        std::string val = arg; to_upper(val);
        for (unsigned i = plask::INTERPOLATION_DEFAULT+1; i != plask::__ILLEGAL_INTERPOLATION_METHOD__; ++i) {
            if (val == plask::interpolationMethodNames[i]) return (plask::InterpolationMethod)i;
        }
        throw bad_lexical_cast(typeid(std::string), typeid(plask::InterpolationMethod));
    }

}



#endif  //PLASK__INTERPOLATION_H
