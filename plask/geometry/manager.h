#ifndef PLASK__GEOMETRY_MANAGER_H
#define PLASK__GEOMETRY_MANAGER_H

/** @file
This file includes:
- plask::GeometryManager class.
*/

#include <string>
#include <map>
#include <set>

#include "container.h"

namespace plask {

/**
 * Geometry manager futures:
 * - read/write geometries
 * - reserve and free memory needs by geometry description
 * - allows for access to geomtrie elements (also by names)
 */
struct GeometryManager {

	///Store pointers to all elements.
	std::set<GeometryElement*> elements;

	///Allow to access to path hints by name.
	std::map<std::string, PathHints*> pathHints;
	
	///Allow to access to elements by name.
	std::map<std::string, GeometryElement*> namedElements;

    GeometryManager();
    
    ///Delete all elements.
    ~GeometryManager();
};

}	// namespace plask

#endif // PLASK__GEOMETRY_MANAGER_H
