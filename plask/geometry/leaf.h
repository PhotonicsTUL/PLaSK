#ifndef PLASK__GEOMETRY_LEAF_H
#define PLASK__GEOMETRY_LEAF_H

/** @file
This file includes geometry elements leafs classes.
*/

#include "element.h"

namespace plask {

/**
For dim equals:
- 2 - rectangle
- 3 - cuboid
*/
template <int dim>
struct Box: public GeometryElementLeaf<dim> {

	typedef typename GeometryElementLeaf<dim>::Vec Vec;
	typedef typename GeometryElementLeaf<dim>::Rect Rect;

	Vec size;
	
	Box(const Vec& size, std::shared_ptr<Material> material): GeometryElementLeaf<dim>(material), size(size) {}
	
	virtual Rect getBoundingBox() {
		return Rect(Primitive<dim>::ZERO_VEC, size);
	}
	
	virtual bool inside(const Vec& p) const {
		return getBoundingBox().inside(p);
	}
	
	virtual bool intersect(const Rect& area) const {
		return getBoundingBox().intersect(area);
	}

};

}	// namespace plask

#endif // PLASK__GEOMETRY_LEAF_H
