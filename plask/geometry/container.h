#ifndef PLASK__GEOMETRY_CONTAINER_H
#define PLASK__GEOMETRY_CONTAINER_H

#include <map>
#include "element.h"

namespace plask {

/**
Represent hints for path finder.

Hints are used to to find unique path for all GeometryElement pairs,
even if one of the pair element is inserted to geometry graph in more than one place.

Each hint allow to choose one child for geometry element container and it is a pair:
geometry element container -> element in container.
*/
struct PathHints {

	///Type for map: geometry element container -> element in container
	typedef std::map<GeometryElement*, GeometryElement*> HintMap;
	
	///Pair type: geometry element container -> element in container
	typedef HintMap::value_type Hint;

	///Hints map.
	HintMap hintFor;

	void addHint(const Hint& hint);
	
	PathHints& operator+=(const Hint& hint) { addHint(hint); return *this; }
	
	void addHint(GeometryElement* container, GeometryElement* child);
	
	/**
	 * Get child for given container.
	 * @return child for given container or @c nullptr if there is no hint for given container
	 */
	GeometryElement* getChild(GeometryElement* container) const;

};


}	// namespace plask

#endif // PLASK__GEOMETRY_CONTAINER_H
