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
 * Geometry manager features:
 * - read/write geometries
 * - reserve and free memory needed by geometry structure
 * - allow access to geometry elements (also by names)
 */
struct GeometryManager {

	/// Store pointers to all elements.
	std::set<GeometryElement*> elements;

	/// Allow to access path hints by name.
	std::map<std::string, PathHints*> pathHints;

	/// Allow to access elements by name.
	std::map<std::string, GeometryElement*> namedElements;

    GeometryManager();

    ///Delete all elements.
    ~GeometryManager();
};

}	// namespace plask

#endif // PLASK__GEOMETRY_MANAGER_H
