#include "rectilinear.h"

namespace plask {

void RectilinearMesh1d::addNode(double new_node_cord) {
	auto where = std::lower_bound(nodes.begin(), nodes.end(), new_node_cord);
	if (where == nodes.end())
		nodes.push_back(new_node_cord);
	else
		if (*where != new_node_cord)	//if node not already included
			nodes.insert(where, new_node_cord);
}

}	//namespace plask
