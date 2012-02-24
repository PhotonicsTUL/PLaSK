#include "element.h"

namespace plask {

GeometryElement::~GeometryElement() {
    changed(Event(this, Event::DELETE));
}

void GeometryElement::ensureCanHasAsParent(GeometryElement& potential_parent) {
    if (isInSubtree(potential_parent))
        throw CyclicReferenceException();
}

}   // namespace plask
