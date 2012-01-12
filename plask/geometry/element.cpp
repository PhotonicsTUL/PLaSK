#include "element.h"

namespace plask {

void GeometryElement::ensureCanHasAsParent(GeometryElement& potential_parent) {
    if (isInSubtree(potential_parent))
        throw CyclicReferenceException();
}

}   // namespace plask
