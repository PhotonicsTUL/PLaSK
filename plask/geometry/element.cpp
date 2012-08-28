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

void GeometryElement::WriteXMLCallback::prerareToAutonaming(const GeometryElement &subtree_root) {
    subtree_root.forEachRealElementInSubtree([&](const GeometryElement& e) { return ++this->counts[&e] == 1; });
}

std::string GeometryElement::WriteXMLCallback::getName(const GeometryElement&, AxisNames&) const {
    return std::string();
}

std::vector<std::string> GeometryElement::WriteXMLCallback::getPathNames(const GeometryElement&, const GeometryElement&, std::size_t) const {
    return std::vector<std::string>();
}

XMLWriter::Element GeometryElement::WriteXMLCallback::makeTag(XMLElement &parent_tag, const GeometryElement &element, AxisNames &axesNames) {
    auto saved_name = names_of_saved.find(&element);
    if (saved_name != names_of_saved.end()) {
        XMLWriter::Element ref(parent_tag, "ref");
        ref.attr("name", saved_name->second);
        return ref;
    }
    XMLWriter::Element tag(parent_tag, element.getTypeName());
    AxisNames newAxesNames = axesNames;
    std::string name = getName(element, newAxesNames);
    if (name.empty()) { //check if auto-name should be constructed
        auto c = counts.find(&element);
        if (c != counts.end() && c->second > 1) { //only for non-unique elements
            name += "#";
            name += boost::lexical_cast<std::string>(nextAutoName);
            ++nextAutoName;
        }
    }
    if (!name.empty()) {
        tag.attr("name", name);
        names_of_saved[&element] = name;
    }
    if (axesNames != newAxesNames) {
        axesNames = std::move(newAxesNames);
        tag.attr("axes", axesNames.str());
    }
    return tag;
}

XMLElement GeometryElement::WriteXMLCallback::makeChildTag(XMLElement& container_tag, const GeometryElement& container, std::size_t index_of_child_in_parent) const {
    XMLElement tag(container_tag, "child");
    //TODO get paths
    return tag;
}

GeometryElement::~GeometryElement() {
    fireChanged(Event::DELETE);
}

void GeometryElement::writeXML(XMLWriter::Element& parent_xml_element, WriteXMLCallback& write_cb, AxisNames axes) const {
    XMLWriter::Element tag = write_cb.makeTag(parent_xml_element, *this, axes);
    if (WriteXMLCallback::isRef(tag)) return;
    writeXMLAttr(tag, axes);
    const std::size_t child_count = getRealChildrenCount();
    for (std::size_t i = 0; i < child_count; ++i)
        getRealChildAt(i)->writeXML(tag, write_cb, axes);
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

std::vector<shared_ptr<const GeometryElement>> GeometryElement::Subtree::getLastPath() const {
    std::vector< shared_ptr<const GeometryElement> > result;
    if (empty()) return result;
    const GeometryElement::Subtree* path_nodes = this;
    while (true) {
        result.push_back(path_nodes->element);
        if (path_nodes->children.empty()) break;
        path_nodes = &(path_nodes->children.back());
    }
return result;
}

void GeometryElement::ensureCanHasAsParent(const GeometryElement& potential_parent) const {
    if (isInSubtree(potential_parent))
        throw CyclicReferenceException();
}

void GeometryElement::writeXMLAttr(XMLWriter::Element& dest_xml_element, const AxisNames& axes) const {
    //do nothing
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

void GeometryElement::forEachRealElementInSubtree(std::function<bool (const GeometryElement &)> callback) const {
    if (!callback(*this)) return;
    std::size_t size = getRealChildrenCount();
    for (std::size_t i = 0; i < size; ++i) getRealChildAt(i)->forEachRealElementInSubtree(callback);
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
