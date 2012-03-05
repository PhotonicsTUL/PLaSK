#include "element.h"

namespace plask {

GeometryElement::~GeometryElement() {
    changed(Event(*this, Event::DELETE));
}

bool GeometryElement::Subtree::isWithBranches() const {
    const std::vector<Subtree>* c = &children;
    while (!c->empty()) {
        if (c->size() > 1) return true;
        c = &((*c)[0].children);
    }
    return false;
}

std::vector< shared_ptr<const GeometryElement> > GeometryElement::Subtree::toLinearPath() const {
    std::vector< shared_ptr<const GeometryElement> > result;
    if (empty()) return result;
    const GeometryElement::Subtree* path_nodes = this;
    while (true) {
        if (path_nodes->children.size() > 1) throw Exception("There are more than one path.");
        result.push_back(path_nodes->element);
        if (path_nodes->children.empty()) break;
        path_nodes = &(path_nodes->children[0]);
    }
    return result;
}

void GeometryElement::ensureCanHasAsParent(GeometryElement& potential_parent) {
    if (isInSubtree(potential_parent))
        throw CyclicReferenceException();
}

}   // namespace plask
