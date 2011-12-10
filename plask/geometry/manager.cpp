#include "manager.h"

namespace plask {

GeometryManager::GeometryManager() {
}

GeometryManager::~GeometryManager() {
	for (GeometryElement* e: elements) delete e;
}

}	// namespace plask
