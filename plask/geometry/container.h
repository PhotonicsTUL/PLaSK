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

Typically, hints are returned by methods which adds new elements to containers.
*/
struct PathHints {

	///Type for map: geometry element container -> element in container
	typedef std::map<GeometryElement*, GeometryElement*> HintMap;
	
	///Pair type: geometry element container -> element in container
	typedef HintMap::value_type Hint;

	///Hints map.
	HintMap hintFor;

	/**
	 * Add hint to hints map. Overwrite if hint for given container already exists.
	 * @param hint hint to add
	 */
	void addHint(const Hint& hint);
	
	/**
	 * Add hint to hints map. Overwrite if hint for given container already exists.
	 * @param hint hint to add
	 */
	PathHints& operator+=(const Hint& hint) { addHint(hint); return *this; }
	
	/**
	 * Add hint to hints map. Overwrite if hint for given container already exists.
	 * @param container, child hint to add
	 */
	void addHint(GeometryElement* container, GeometryElement* child);
	
	/**
	 * Get child for given container.
	 * @return child for given container or @c nullptr if there is no hint for given container
	 */
	GeometryElement* getChild(GeometryElement* container) const;

};


}	// namespace plask

#endif // PLASK__GEOMETRY_CONTAINER_H
