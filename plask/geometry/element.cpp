#include "element.h"

#include "leaf.h"

#include "transform.h"
#include "space.h"

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
    fireChanged(Event::DELETE);
}

template<int DIMS>
shared_ptr< GeometryElementD<DIMS> > GeometryElement::asD() {
    if (getDimensionsCount() != DIMS || isGeometry()) return shared_ptr< GeometryElementD<DIMS> >();
    return static_pointer_cast< GeometryElementD<DIMS> >(shared_from_this());
}

template<int DIMS>
shared_ptr< const GeometryElementD<DIMS> > GeometryElement::asD() const {
    if (getDimensionsCount() != DIMS || isGeometry()) return shared_ptr< const GeometryElementD<DIMS> >();
    return static_pointer_cast< const GeometryElementD<DIMS> >(shared_from_this());
}

template shared_ptr< GeometryElementD<2> > GeometryElement::asD<2>();
template shared_ptr< GeometryElementD<3> > GeometryElement::asD<3>();
template shared_ptr< const GeometryElementD<2> > GeometryElement::asD<2>() const;
template shared_ptr< const GeometryElementD<3> > GeometryElement::asD<3>() const;

shared_ptr<Geometry> GeometryElement::asGeometry() {
    return isGeometry() ? static_pointer_cast<Geometry>(shared_from_this()) : shared_ptr<Geometry>();
}

shared_ptr<const Geometry> GeometryElement::asGeometry() const {
    return isGeometry() ? static_pointer_cast<const Geometry>(shared_from_this()) : shared_ptr<const Geometry>();
}

bool GeometryElement::isInSubtree(const GeometryElement &el) const {
    if (&el == this) return true;
    std::size_t c = getRealChildrenCount();
    for (std::size_t i = 0; i < c; ++i)
        if (getRealChildAt(i)->isInSubtree(el))
            return true;
    return false;
}

bool GeometryElement::Subtree::hasBranches() const {
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
        if (path_nodes->children.size() > 1) throw NotUniqueElementException("There is more than one path in the subtree.");
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

std::size_t GeometryElement::getRealChildrenCount() const {
    return getChildrenCount();
}

shared_ptr<GeometryElement> GeometryElement::getRealChildAt(std::size_t child_nr) const {
    return getChildAt(child_nr);
}

void GeometryElement::removeAtUnsafe(std::size_t) {
    throw NotImplemented("removeAtUnsafe(std::size_t)");
}

// --- GeometryElementD ---

template <int dimensions>
shared_ptr<Translation<dimensions>>
GeometryElementD<dimensions>::getElementInThisCoordinates(const shared_ptr<GeometryElementD<dimensions>>& element, const PathHints* path) const {
    auto trans_vec = getElementPositions(*element, path);
    if (trans_vec.size() != 1 || std::isnan(trans_vec[0].components[0]))
        shared_ptr<Translation<dimensions>>();
    return make_shared<Translation<dimensions>>(element, trans_vec[0]);
}

template class GeometryElementD<2>;
template class GeometryElementD<3>;


}   // namespace plask
