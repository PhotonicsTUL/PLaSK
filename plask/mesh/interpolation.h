namespace plask {

/**
Supported interpolation methods.
*/
enum InterpolationMethod {
    DEFAULT = 0,
    LINEAR = 1
};

/**
Specialization of this class are used for interpolation and can depend from source mesh type, data type and method.
*/
struct <typename SrcMeshT, typename DataT, InterpolationMethod method>
InterpolationAlgorithm {
    static void interpolate(SrcMeshT& src_mesh, std::vector<T>& src_vec, SrcMeshT& dst_mesh, std::vector<T>& dst_vec) throw (NotImplemented) {
        throw NotImplemented(TODO);
        //TODO iterate over dst_mesh and call InterpolationAlgorithmForPoint
    }
};

/**
Interpolate values (@a src_vec) from one mesh (@a src_mesh) to another one (@a dst_mesh) using given interpolation method.
@param src_mesh, src_vec source
@param dst_mesh destination mesh
@param method interpolation method to use
@throw NotImplemented if given interpolation method is not implemented for used source mesh type
@throw NotSuchInterpolationMethod if given interpolation method is bad
*/
template <typename SrcMeshT, typename DataT>
inline std::shared_ptr<std::vector<DataT>>
interpolate(SrcMeshT& src_mesh, std::shared_ptr<std::vector<DataT>>& src_vec, Mesh& dst_mesh, InterpolationMethod method) throw (NotImplemented, NoSuchInterpolationMethod) {
    if (&src_mesh == &dst_mesh)        // meshs are identicall,
        return src_vec;                // just return src_vec
    std::shared_ptr<std::vector<DataT>> result(new std::vector);
    switch (method) {
        case DEFAULT:
            InterpolationAlgorithm<SrcMeshT, DataT, DEFAULT>::interpolate(src_mesh, *src_vec, dst_mesh, *result);
            break;
        case LINEAR:
            InterpolationAlgorithm<SrcMeshT, DataT, LINEAR>::interpolate(src_mesh, *src_vec, dst_mesh, *result);
            break;
        default:
            throw NoSuchInterpolationMethod();
    }
    return result;
}

} // namespace plask
