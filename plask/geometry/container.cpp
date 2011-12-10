#include "container.h"
#include "../utils/stl.h"

namespace plask {

void PathHints::addHint(const Hint& hint) {
	hintFor.insert(hint);
}

void PathHints::addHint(GeometryElement* container, GeometryElement* child) {
	hintFor[container] = child;
}

GeometryElement* PathHints::getChild(GeometryElement* container) const {
	return map_find(hintFor, container);
}

}	// namespace plask
