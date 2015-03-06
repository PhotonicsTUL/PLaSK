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
@a src_mesh, @a src_vec, @a dst_mesh, interpolation method, and interpolation flags that specify behavior outside of the source mesh.

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
#include "../lazydata.h"
#include "../log/log.h"
#include <plask/geometry/space.h>

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
inline InterpolationMethod getInterpolationMethod(InterpolationMethod method) {
    return (method == INTERPOLATION_DEFAULT)? default_method : method;
}

/**
 * Interpolation flags that give information how to interpolate fields on periodic and symmetric geometries.
 */
class InterpolationFlags {
    unsigned char sym[3];   ///< Symmetry along axes (bit[0] for symmetry bits[1-3] are used for positive and negative reflection of vector components)
    unsigned char per;      ///< Information about periodicity (in bits)
    double lo[3], hi[3];    ///< Limits of the axes (required for periodic interpolation)

public:

    enum class Symmetry: unsigned char {
        NO = 0,
        POSITIVE = 1,
        PP = 1,
        NP = 3,
        PN = 5,
        NN = 7,
        PPP = 1,
        NPP = 3,
        PNP = 5,
        NNP = 7,
        PPN = 9,
        NPN = 11,
        PNN = 13,
        NNN = 15,
        NEGATIVE = 15,
    };

    InterpolationFlags(): sym{0,0,0}, per(0), lo{0.,0.,0.} , hi{0., 0., 0.}  {}

    InterpolationFlags(const shared_ptr<GeometryD<2>>& geometry, Symmetry sym0=Symmetry::POSITIVE, Symmetry sym1=Symmetry::POSITIVE):
        sym{geometry->isSymmetric(Geometry::DIRECTION_TRAN)? (unsigned char)sym0 : 0, geometry->isSymmetric(Geometry::DIRECTION_VERT)? (unsigned char)sym1 : 0},
        per((geometry->isPeriodic(Geometry::DIRECTION_TRAN)? 1 : 0) + (geometry->isPeriodic(Geometry::DIRECTION_VERT)? 2 : 0)),
        lo{geometry->getChildBoundingBox().left(), geometry->getChildBoundingBox().bottom(), 0.},
        hi{geometry->getChildBoundingBox().right(), geometry->getChildBoundingBox().top(), 0.}
    {
        if (sym[0] && lo[0] < 0. && hi[0] > 0.) throw Exception("interpolation: Symmetric geometry spans at both sides of transverse axis");
        if (sym[1] && lo[1] < 0. && hi[1] > 0.) throw Exception("interpolation: Symmetric geometry spans at both sides of vertical axis");
    }

    InterpolationFlags(const shared_ptr<GeometryD<3>>& geometry, Symmetry sym0=Symmetry::POSITIVE, Symmetry sym1=Symmetry::POSITIVE, Symmetry sym2=Symmetry::POSITIVE):
        sym{geometry->isSymmetric(Geometry::DIRECTION_LONG)? (unsigned char)sym0 : 0, geometry->isSymmetric(Geometry::DIRECTION_TRAN)? (unsigned char)sym1 : 0, geometry->isSymmetric(Geometry::DIRECTION_VERT)? (unsigned char)sym2 : 0},
        per((geometry->isPeriodic(Geometry::DIRECTION_LONG)? 1 : 0) + (geometry->isPeriodic(Geometry::DIRECTION_TRAN)? 2 : 0) + (geometry->isPeriodic(Geometry::DIRECTION_VERT)? 4 : 0)),
        lo{geometry->getChildBoundingBox().back(), geometry->getChildBoundingBox().left(), geometry->getChildBoundingBox().bottom()},
        hi{geometry->getChildBoundingBox().front(), geometry->getChildBoundingBox().right(), geometry->getChildBoundingBox().top()}
    {
        if (sym[0] && lo[0] < 0. && hi[0] > 0.) throw Exception("interpolation: Symmetric geometry spans at both sides of longitudinal axis");
        if (sym[1] && lo[1] < 0. && hi[1] > 0.) throw Exception("interpolation: Symmetric geometry spans at both sides of transverse axis");
        if (sym[2] && lo[2] < 0. && hi[2] > 0.) throw Exception("interpolation: Symmetric geometry spans at both sides of vertical axis");
    }

    bool symmetric(int axis) const { return sym[axis]; }

    template <int axis>
    bool symmetric() const { return sym[axis]; }

    bool periodic(int axis) const { return per & (1 << axis); }

    template <int axis>
    bool periodic() const { return per & (1 << axis); }

    double low(int axis) const {
        if (sym[axis]) return min(lo[axis], -hi[axis]);
        else return lo[axis];
    }

    template <int axis>
    double low() const {
        if (sym[axis]) return min(lo[axis], -hi[axis]);
        else return lo[axis];
    }

    double high(int axis) const {
        if (sym[axis]) return max(lo[axis], hi[axis]);
        else return hi[axis];
    }

    template <int axis>
    double high() const {
        if (sym[axis]) return max(lo[axis], hi[axis]);
        else return hi[axis];
    }

    template <int dim>
    Vec<dim> wrap(Vec<dim> pos) const {
        for (int i = 0; i != dim; ++i) {
            double d = hi[i] - lo[i];
            if (periodic(i)) {
                if (sym[i]) {
                    pos[i] = std::fmod(pos[i], 2.*d);
                    if (pos[i] > d) pos[i] -= 2.*d;
                    else if (pos[i] < -d) pos[i] += 2*d;
                    if (lo[i] < 0) pos[i] = - pos[i];
                } else {
                    pos[i] = std::fmod(pos[i]-lo[i], d);
                    pos[i] += (pos[i] >= 0)? lo[i] : hi[i];
                }
            } else if (sym[i]) {
                if (lo[i] >= 0) pos[i] = abs(pos[i]);
                else pos[i] = - abs(pos[i]);
            }
        }
        return pos;
    }

    template <int dim, typename DataT>
    Vec<dim,DataT> reflect(int ax, Vec<dim,DataT> vec) const {
        for (int i = 0; i != dim; ++i) if (sym[ax] & (2 << i)) vec[i] = -vec[i];
        return vec;
    }

    template <typename DataT>
    DataT reflect(int ax, DataT val) const {
        if (sym[ax] & 14) return -val;
        else return val;
    }

    template <int dim, typename DataT>
    DataT postprocess(Vec<dim> pos, DataT data) const {
        for (int i = 0; i != dim; ++i) {
            if (sym[i]) {
                if (periodic(i)) {
                    double d = hi[i] - lo[i];
                    pos[i] = std::fmod(pos[i], 2.*d);
                    if (pos[i] > d || (pos[i] < 0. && pos[i] > -d)) data = reflect(i, data);
                } else {
                    if (lo[i] >= 0) { if (pos[i] < 0.) data = reflect(i, data); }
                    else { if (pos[i] > 0.) data = reflect(i, data); }
                }
            }
        }
        return data;
    }
};


/**
 * Specialization of this class are used for interpolation and can depend on source mesh type,
 * data type and the interpolation method.
 * @see @ref interpolation_write
 */
template <typename SrcMeshT, typename SrcT, typename DstT, InterpolationMethod method>
struct InterpolationAlgorithm
{
    static LazyData<DstT> interpolate(const shared_ptr<const SrcMeshT>& src_mesh, const DataVector<const SrcT>&,
                                      const shared_ptr<const MeshD<SrcMeshT::DIM>>&, const InterpolationFlags&) {
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
    static LazyData<DstT> interpolate(const shared_ptr<const SrcMeshT>&, const DataVector<const SrcT>&,
                                      const shared_ptr<const MeshD<SrcMeshT::DIM>>&, const InterpolationFlags& flags) {
        throw CriticalException("interpolate(...) called for INTERPOLATION_DEFAULT method. Contact solver author to fix this issue."
#ifndef NDEBUG
                                "\n\nINFO FOR SOLVER AUTHOR: To avoid this error use 'getInterpolationMethod<YOUR_DEFAULT_METHOD>(interpolation_method) in C++ code of the provider in your solver.\n"
#endif
                               );
    }
};

#ifndef DOXYGEN
// The following structures are solely used for metaprogramming
template <typename SrcMeshT, typename SrcT, typename DstT, int iter>
struct __InterpolateMeta__
{
    inline static LazyData<typename std::remove_const<DstT>::type> interpolate(
            const shared_ptr<const SrcMeshT>& src_mesh, const DataVector<const SrcT>& src_vec,
            const shared_ptr<const MeshD<SrcMeshT::DIM>>& dst_mesh, InterpolationMethod method, const InterpolationFlags& flags) {
        if (int(method) == iter)
            return InterpolationAlgorithm<SrcMeshT, SrcT, typename std::remove_const<DstT>::type, (InterpolationMethod)iter>
                    ::interpolate(src_mesh, DataVector<const SrcT>(src_vec), dst_mesh, flags);
        else
            return __InterpolateMeta__<SrcMeshT, SrcT, DstT, iter+1>::interpolate(src_mesh, src_vec, dst_mesh, method, flags);
    }
};
template <typename SrcMeshT, typename SrcT, typename DstT>
struct __InterpolateMeta__<SrcMeshT, SrcT, DstT, __ILLEGAL_INTERPOLATION_METHOD__>
{
    inline static LazyData<typename std::remove_const<DstT>::type> interpolate(const shared_ptr<const SrcMeshT>&, const DataVector<const SrcT>&,
                                   const shared_ptr<const MeshD<SrcMeshT::DIM>>&, InterpolationMethod, const InterpolationFlags&) {
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
                                        const InterpolationFlags& flags=InterpolationFlags(),
                                        bool verbose=true)
{
    if (src_mesh->size() != src_vec.size())
        throw BadMesh("interpolate", "Mesh size (%2%) and values size (%1%) do not match", src_vec.size(), src_mesh->size());
    if (src_mesh == dst_mesh) return new LazyDataFromVectorImpl<typename std::remove_const<DstT>::type>(src_vec); // meshes are identical, so just return src_vec
    if (verbose) writelog(LOG_DEBUG, "interpolate: Running %1% interpolation", interpolationMethodNames[method]);
    return __InterpolateMeta__<SrcMeshT, SrcT, DstT, 0>::interpolate(src_mesh, src_vec, dst_mesh, method, flags);
}

template <typename SrcMeshT, typename SrcT, typename DstT=SrcT, typename DstMeshT>
LazyData<typename std::remove_const<DstT>::type> interpolate(
                            shared_ptr<SrcMeshT> src_mesh,
                            DataVector<SrcT> src_vec,
                            shared_ptr<DstMeshT> dst_mesh,
                            InterpolationMethod method=INTERPOLATION_DEFAULT,
                            const InterpolationFlags& flags=InterpolationFlags(),
                            bool verbose=true)
{
    return interpolate(shared_ptr<const SrcMeshT>(src_mesh), DataVector<const SrcT>(src_vec),
                       shared_ptr<const MeshD<SrcMeshT::DIM>>(dst_mesh), method, flags, verbose);
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
template <typename DstT, typename SrcMeshType, typename SrcT = DstT>
struct InterpolatedLazyDataImpl: public LazyDataImpl<DstT> {

    shared_ptr<const SrcMeshType> src_mesh;
    shared_ptr<const MeshD<SrcMeshType::DIM>> dst_mesh;
    DataVector<const SrcT> src_vec;
    InterpolationFlags flags;

    InterpolatedLazyDataImpl(const shared_ptr<const SrcMeshType>& src_mesh, const DataVector<const SrcT>& src_vec,
                             const shared_ptr<const MeshD<SrcMeshType::DIM>>& dst_mesh, const InterpolationFlags& flags)
        : src_mesh(src_mesh), dst_mesh(dst_mesh), src_vec(src_vec), flags(flags) {}

    virtual std::size_t size() const override { return dst_mesh->size(); }

};

/**
 * Implementation of InterpolatedLazyDataImpl which call src_mesh.interpolateLinear(src_vec, dst_mesh).
 *
 * So it can be used if SrcMeshType has such interpolateLinear method.
 */
template <typename DstT, typename SrcMeshType, typename SrcT = DstT>
struct LinearInterpolatedLazyDataImpl: public InterpolatedLazyDataImpl<DstT, SrcMeshType, SrcT> {

    LinearInterpolatedLazyDataImpl(shared_ptr<const SrcMeshType> src_mesh, const DataVector<const SrcT>& src_vec,
                                   shared_ptr<const MeshD<SrcMeshType::DIM>> dst_mesh, const InterpolationFlags& flags):
        InterpolatedLazyDataImpl<DstT, SrcMeshType>(src_mesh, src_vec, dst_mesh, flags) {}

    virtual DstT at(std::size_t index) const override {
        return this->src_mesh->interpolateLinear(this->src_vec, this->dst_mesh->at(index), this->flags);
    }

};

/**
 * Implementation of InterpolatedLazyDataImpl which call src_mesh.interpolateNearestNeighbor(src_vec, dst_mesh).
 *
 * So it can be used if SrcMeshType has such interpolateLinear method.
 */
template <typename DstT, typename SrcMeshType, typename SrcT=DstT>
struct NearestNeighborInterpolatedLazyDataImpl: public InterpolatedLazyDataImpl<DstT, SrcMeshType, SrcT> {

    NearestNeighborInterpolatedLazyDataImpl(shared_ptr<const SrcMeshType> src_mesh, const DataVector<const SrcT>& src_vec,
                                            shared_ptr<const MeshD<SrcMeshType::DIM>> dst_mesh, const InterpolationFlags& flags):
        InterpolatedLazyDataImpl<DstT, SrcMeshType, DstT>(src_mesh, src_vec, dst_mesh, flags) {}

    virtual SrcT at(std::size_t index) const override {
        return this->src_mesh->interpolateNearestNeighbor(this->src_vec, this->dst_mesh->at(index), this->flags);
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
