#include "element.h"

namespace plask {

bool GeometryElement::Subtree::isWithBranches() const {
    const std::vector<Subtree>* c = &children;
    while (!c->empty()) {
        if (c->size() > 1) return true;
        c = &((*c)[0].children);
    }
    return false;
}

GeometryElement::~GeometryElement() {
    changed(Event(*this, Event::DELETE));
}

void GeometryElement::ensureCanHasAsParent(GeometryElement& potential_parent) {
    if (isInSubtree(potential_parent))
        throw CyclicReferenceException();
}

}   // namespace plask
