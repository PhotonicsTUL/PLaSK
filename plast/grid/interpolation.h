namespace plast {

/**
Supported interpolation methods.
*/
enum InterpolationMethod {
	linear = 0
};

/**
Specialization of this class are used for interpolation and can depend from source grid type, data type and method.
*/
struct <typename SrcGridT, typename DataT, InterpolationMethod method>
InterpolationAlgorithm {
    static void interpolate(SrcGridT& src_grid, std::vector<T>& src_vec, SrcGridT::BaseClass& dst_grid, std::vector<T>& dst_vec) throw (NotImplemented) {
	throw NotImplemented(TODO);
	//TODO iterate over dst_grid and call InterpolationAlgorithmForPoint
    }
};

/**
Interpolate values (@a src_vec) from one grid (@a src_grid) to another one (@a dst_grid) using given interpolation method.
@param src_grid, src_vec source
@param dst_grid destination grid
@param method interpolation method to use
@throw NotImplemented if given interpolation method is not implemented for used source grid type
@throw NotSuchInterpolationMethod if given interpolation method is bad
*/
template <typename SrcGridT, typename DataT>
inline std::shared_ptr<std::vector<DataT>>
interpolate(SrcGridT& src_grid, std::shared_ptr<std::vector<DataT>>& src_vec, SrcGridT::BaseClass& dst_grid, InterpolationMethod method) throw (NotImplemented, NoSuchInterpolationMethod) {
    if (&src_grid == &dst_grid)	//grids are identicall,
	return src_vec;		//just return src_vec
    std::shared_ptr<std::vector<DataT>> result(new std::vector);
    switch (method) {
	case linear:
	    InterpolationAlgorithm<SrcGridT, DataT, linear>(src_grid, *src_vec, dst_grid, *result);
	    break;
	default:
	    throw NoSuchInterpolationMethod();
    }
    return result;
}

}	//namespace plast