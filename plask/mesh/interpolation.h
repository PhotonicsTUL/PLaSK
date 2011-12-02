#ifndef PLASK__INTERPOLATION_H
#define PLASK__INTERPOLATION_H

#include <typeinfo>  // for 'typeid'

#include "mesh.h"
#include "plask/exceptions.h"

namespace plask {

/**
 * Supported interpolation methods.
 * @see @ref meshes_interpolation
 */
enum InterpolationMethod {
    DEFAULT = 0,        ///< default interpolation (depends on source mesh)
    LINEAR = 1,         ///< linear interpolation
    SPLINE = 2,         ///< spline interpolation
    //...add new interpolation algoritms here...
    __ILLEGAL_INTERPOLATION_METHOD__  // necessary for metaprogram loop
};

static const char* InterpolationMethodNames[] = { "DEFAULT", "LINEAR", "SPLINE" /*attach new interpolation algoritm names here*/};

/**
 * Specialization of this class are used for interpolation and can depend on source mesh type,
 * data type and the interpolation method.
 * @see @ref interpolation_write
 */
template <typename SrcMeshT, typename DataT, InterpolationMethod method>
struct InterpolationAlgorithm
{
    static void interpolate(SrcMeshT& src_mesh, const std::vector<DataT>& src_vec, const Mesh<typename SrcMeshT::Space>& dst_mesh, std::vector<DataT>& dst_vec) {
        std::string msg = "interpolate (source mesh type: ";
        msg += typeid(src_mesh).name();
        msg += ", interpolation method: ";
        msg += InterpolationMethodNames[method];
        msg += ")";
        throw NotImplemented(msg);
        //TODO iterate over dst_mesh and call InterpolationAlgorithmForPoint
    }
};

// The following structures are solely used for metaprogramming
template <typename SrcMeshT, typename DataT, int iter>
struct __InterpolateMeta__
{
    inline static void interpolate(SrcMeshT& src_mesh, const std::vector<DataT>& src_vec,
                Mesh<typename SrcMeshT::Space>& dst_mesh, std::vector<DataT>& dst_vec, InterpolationMethod method) {
        if (int(method) == iter)
            InterpolationAlgorithm<SrcMeshT, DataT, (InterpolationMethod)iter>::interpolate(src_mesh, src_vec, dst_mesh, dst_vec);
        else
            __InterpolateMeta__<SrcMeshT, DataT, iter+1>::interpolate(src_mesh, src_vec, dst_mesh, dst_vec, method);
    }
};
template <typename SrcMeshT, typename DataT>
struct __InterpolateMeta__<SrcMeshT, DataT, __ILLEGAL_INTERPOLATION_METHOD__>
{
    inline static void interpolate(SrcMeshT& src_mesh, const std::vector<DataT>& src_vec,
                Mesh<typename SrcMeshT::Space>& dst_mesh, std::vector<DataT>& dst_vec, InterpolationMethod method) {
        throw CriticalException("No such interpolation method.");
    }
};


/**
 * Calculate (interpolate when needed) a field of some physical properties in requested points of (@a dst_mesh)
 * if values of this field in points of (@a src_mesh) are known.
 * @param src_mesh set of points in which fields values are known
 * @param src_vec vector of known field values in points described by @a sec_mesh
 * @param dst_mesh requested set of points, in which the field values should be calculated (interpolated)
 * @param method interpolation method to use
 * @return vector of the field values in points described by @a dst_mesh, can be equal to @a src_vec
 *         if @a src_mesh and @a dst_mesh are the same mesh
 * @throw NotImplemented if given interpolation method is not implemented for used source mesh type
 * @throw CriticalException if given interpolation method is not valid
 * @see @ref meshes_interpolation
 */
template <typename SrcMeshT, typename DataT>
inline std::shared_ptr<const std::vector<DataT>>
interpolate(SrcMeshT& src_mesh, std::shared_ptr<const std::vector<DataT>>& src_vec,
            Mesh<typename SrcMeshT::Space>& dst_mesh, InterpolationMethod method = DEFAULT,
            std::shared_ptr<const std::vector<DataT>> dst_vec = (std::shared_ptr<const std::vector<DataT>>)nullptr) {

    if (&src_mesh == &dst_mesh) return src_vec; // meshes are identical, so just return src_vec

    std::shared_ptr<std::vector<DataT>> result;
    if (dst_vec == nullptr) result = std::shared_ptr<std::vector<DataT>>(new std::vector<DataT>);
    else result = (std::shared_ptr<std::vector<DataT>>&)dst_vec;

    result->resize(dst_mesh.size());
    __InterpolateMeta__<SrcMeshT, DataT, 0>::interpolate(src_mesh, *src_vec, dst_mesh, *result, method);

    return result;
}


} // namespace plask

#endif  //PLASK__INTERPOLATION_H
