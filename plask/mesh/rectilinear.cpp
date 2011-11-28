#include "rectilinear.h"

namespace plask {

RectilinearMesh1d::const_iterator RectilinearMesh1d::find(double to_find) const {
    return std::lower_bound(points.begin(), points.end(), to_find);
}

void RectilinearMesh1d::addPoint(double new_node_cord) {
	auto where = std::lower_bound(points.begin(), points.end(), new_node_cord);
	if (where == points.end())
		points.push_back(new_node_cord);
	else
		if (*where != new_node_cord)	//if node not already included
			points.insert(where, new_node_cord);
}

}	//namespace plask
