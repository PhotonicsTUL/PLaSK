
enum InterpolationMethod {
	linear = 0
};


template <SrcGridT, T>
std::shared_ptr<std::vector<T>> interpolate(SrcGridT& src_grid, std::vector<T>& src_vec, Grid& dst_grid, InterpolationMethod method);

