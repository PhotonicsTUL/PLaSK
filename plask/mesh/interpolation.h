#ifndef PLASK__INTERPOLATION_H
#define PLASK__INTERPOLATION_H

#include <typeinfo>  //for 'typeid'

struct Mesh;

namespace plask {

/**
Supported interpolation methods.
*/
enum InterpolationMethod {
    DEFAULT = 0,        ///<default interpolation depend from source
    LINEAR = 1          ///<linear interpolation
};

const char* InterpolationMethodNames[] = { "DEFAULT", "LINEAR" };

/**
Specialization of this class are used for interpolation and can depend from source mesh type, data type and method.
@see @ref meshes_write
*/
template <typename SrcMeshT, typename DataT, InterpolationMethod method>
struct InterpolationAlgorithm {
    static void interpolate(SrcMeshT& src_mesh, const std::vector<DataT>& src_vec, const Mesh& dst_mesh, std::vector<DataT>& dst_vec) throw (NotImplemented) {
        std::string msg = "interpolate for source grid type ";
        msg += typeid(src_mesh).name();
        msg += " and interpolation type ";
        msg += InterpolationMethodNames[method];
        throw NotImplemented(msg);
        //TODO iterate over dst_mesh and call InterpolationAlgorithmForPoint
    }
};

/**
Interpolate values (@a src_vec) from one mesh (@a src_mesh) to another one (@a dst_mesh) using given interpolation method.
@param src_mesh, src_vec source
@param dst_mesh destination mesh
@param method interpolation method to use
@return vector with interpolated values
@throw NotImplemented if given interpolation method is not implemented for used source mesh type
@throw CriticalException if given interpolation method is not valid
*/
template <typename SrcMeshT, typename DataT>
inline std::shared_ptr<const std::vector<DataT>>
interpolate(SrcMeshT& src_mesh, std::shared_ptr<const std::vector<DataT>>& src_vec, Mesh& dst_mesh, InterpolationMethod method = DEFAULT)
throw (NotImplemented, CriticalException) {
    if (&src_mesh == &dst_mesh)        // meshs are identicall,
        return src_vec;                // just return src_vec
    std::shared_ptr<std::vector<DataT>> result(new std::vector<DataT>);
    switch (method) {
        case DEFAULT:
            InterpolationAlgorithm<SrcMeshT, DataT, DEFAULT>::interpolate(src_mesh, *src_vec, dst_mesh, *result);
            break;
        case LINEAR:
            InterpolationAlgorithm<SrcMeshT, DataT, LINEAR>::interpolate(src_mesh, *src_vec, dst_mesh, *result);
            break;
        default:
            throw CriticalException("No such interpolation method.");
    }
    return result;
}

} // namespace plask

#endif  //PLASK__INTERPOLATION_H
