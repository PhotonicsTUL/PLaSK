#include "container.h"
#include "../utils/stl.h"

namespace plask {

void PathHints::addHint(const Hint& hint) {
	addHint(hint.first, hint.second);
}

void PathHints::addHint(GeometryElement* container, GeometryElement* child) {
	hintFor[container] = child;
}

GeometryElement* PathHints::getChild(GeometryElement* container) const {
	return map_find(hintFor, container);
}

}	// namespace plask
