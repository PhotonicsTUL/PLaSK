#include "element.h"

#include "leaf.h"

namespace plask {

GeometryElement::CompositeChanger::CompositeChanger(const Changer* changer) {
    changers.push_back(changer);
}

GeometryElement::CompositeChanger& GeometryElement::CompositeChanger::operator()(const Changer* changer) {
    changers.push_back(changer);
    return *this;
}

GeometryElement::CompositeChanger::~CompositeChanger() {
    for (auto c: changers) delete c;
}

bool GeometryElement::CompositeChanger::apply(shared_ptr<const GeometryElement>& to_change, Vec<3, double>* translation) const {
    for (auto c: changers) if (c->apply(to_change, translation)) return true;
    return false;
}

bool GeometryElement::ReplaceChanger::apply(shared_ptr<const GeometryElement>& to_change, Vec<3, double>* translation) const {
    if (to_change != from) return false;
    to_change = to;
    if (translation) *translation = this->translation;
    return true;
}

GeometryElement::ToBlockChanger::ToBlockChanger(const shared_ptr<const GeometryElement>& toChange, const shared_ptr<Material> &material) {
    from = toChange;
    to = changeToBlock(material, from, translation);
}

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

void GeometryElement::ensureCanHasAsParent(const GeometryElement& potential_parent) const {
    if (isInSubtree(potential_parent))
        throw CyclicReferenceException();
}



}   // namespace plask
