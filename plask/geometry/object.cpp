#include "object.h"

#include "leaf.h"

#include "transform.h"
#include "space.h"
#include "path.h"

namespace plask {

GeometryObject::CompositeChanger::CompositeChanger(const Changer* changer) {
    changers.push_back(changer);
}

GeometryObject::CompositeChanger& GeometryObject::CompositeChanger::operator()(const Changer* changer) {
    changers.push_back(changer);
    return *this;
}

GeometryObject::CompositeChanger::~CompositeChanger() {
    for (auto c: changers) delete c;
}

bool GeometryObject::CompositeChanger::apply(shared_ptr<GeometryObject> &to_change, Vec<3, double>* translation) const {
    for (auto c: changers) if (c->apply(to_change, translation)) return true;
    return false;
}

bool GeometryObject::ReplaceChanger::apply(shared_ptr<GeometryObject> &to_change, Vec<3, double>* translation) const {
    if (to_change != from) return false;
    to_change = to;
    if (translation) *translation = this->translation;
    return true;
}

GeometryObject::ToBlockChanger::ToBlockChanger(shared_ptr<const GeometryObject> toChange, shared_ptr<Material> material) {
    from = toChange;
    to = changeToBlock(material, from, translation);
}

bool GeometryObject::DeleteChanger::apply(shared_ptr<GeometryObject>& to_change, Vec<3, double>* translation) const {
    if (to_change != toDel) return false;
    to_change = shared_ptr<GeometryObject>();
    return true;
}

void GeometryObject::WriteXMLCallback::prerareToAutonaming(const GeometryObject &subtree_root) {
    subtree_root.forEachRealObjectInSubtree([&](const GeometryObject& e) { return ++this->counts[&e] == 1; });
}

std::string GeometryObject::WriteXMLCallback::getName(const GeometryObject&, AxisNames&) const {
    return std::string();
}

std::vector<std::string> GeometryObject::WriteXMLCallback::getPathNames(const GeometryObject&, const GeometryObject&, std::size_t) const {
    return std::vector<std::string>();
}

XMLWriter::Element GeometryObject::WriteXMLCallback::makeTag(XMLElement &parent_tag, const GeometryObject &object, AxisNames &axesNames) {
    auto saved_name = names_of_saved.find(&object);
    if (saved_name != names_of_saved.end()) {
        XMLWriter::Element ref(parent_tag, "ref");
        ref.attr("name", saved_name->second);
        return ref;
    }
    XMLWriter::Element tag(parent_tag, object.getTypeName());
    AxisNames newAxesNames = axesNames;
    std::string name = getName(object, newAxesNames);
    if (name.empty()) { //check if auto-name should be constructed
        auto c = counts.find(&object);
        if (c != counts.end() && c->second > 1) { //only for non-unique objects
            name += "#";
            name += boost::lexical_cast<std::string>(nextAutoName);
            ++nextAutoName;
        }
    }
    if (!name.empty()) {
        tag.attr("name", name);
        names_of_saved[&object] = name;
    }
    if (axesNames != newAxesNames) {
        axesNames = std::move(newAxesNames);
        tag.attr("axes", axesNames.str());
    }
    return tag;
}

XMLElement GeometryObject::WriteXMLCallback::makeChildTag(XMLElement& container_tag, const GeometryObject& container, std::size_t index_of_child_in_parent) const {
    XMLElement tag(container_tag, "child");
    //TODO get paths
    return tag;
}

GeometryObject::~GeometryObject() {
    fireChanged(Event::DELETE);
}

void GeometryObject::writeXML(XMLWriter::Element& parent_xml_object, WriteXMLCallback& write_cb, AxisNames axes) const {
    XMLWriter::Element tag = write_cb.makeTag(parent_xml_object, *this, axes);
    if (WriteXMLCallback::isRef(tag)) return;
    writeXMLAttr(tag, axes);
    const std::size_t child_count = getRealChildrenCount();
    for (std::size_t i = 0; i < child_count; ++i)
        getRealChildAt(i)->writeXML(tag, write_cb, axes);
}

template<int DIMS>
shared_ptr< GeometryObjectD<DIMS> > GeometryObject::asD() {
    if (getDimensionsCount() != DIMS || isGeometry()) return shared_ptr< GeometryObjectD<DIMS> >();
    return static_pointer_cast< GeometryObjectD<DIMS> >(shared_from_this());
}

template<int DIMS>
shared_ptr< const GeometryObjectD<DIMS> > GeometryObject::asD() const {
    if (getDimensionsCount() != DIMS || isGeometry()) return shared_ptr< const GeometryObjectD<DIMS> >();
    return static_pointer_cast< const GeometryObjectD<DIMS> >(shared_from_this());
}

template shared_ptr< GeometryObjectD<2> > GeometryObject::asD<2>();
template shared_ptr< GeometryObjectD<3> > GeometryObject::asD<3>();
template shared_ptr< const GeometryObjectD<2> > GeometryObject::asD<2>() const;
template shared_ptr< const GeometryObjectD<3> > GeometryObject::asD<3>() const;

shared_ptr<Geometry> GeometryObject::asGeometry() {
    return isGeometry() ? static_pointer_cast<Geometry>(shared_from_this()) : shared_ptr<Geometry>();
}

shared_ptr<const Geometry> GeometryObject::asGeometry() const {
    return isGeometry() ? static_pointer_cast<const Geometry>(shared_from_this()) : shared_ptr<const Geometry>();
}

bool GeometryObject::isInSubtree(const GeometryObject &el) const {
    if (&el == this) return true;
    std::size_t c = getRealChildrenCount();
    for (std::size_t i = 0; i < c; ++i)
        if (getRealChildAt(i)->isInSubtree(el))
            return true;
    return false;
}

bool GeometryObject::Subtree::hasBranches() const {
    const std::vector<Subtree>* c = &children;
    while (!c->empty()) {
        if (c->size() > 1) return true;
        c = &((*c)[0].children);
    }
    return false;
}

Path GeometryObject::Subtree::toLinearPath() const {
    std::vector< shared_ptr<const GeometryObject> > result;
    if (empty()) return result;
    const GeometryObject::Subtree* path_nodes = this;
    while (true) {
        if (path_nodes->children.size() > 1) throw NotUniqueObjectException("There is more than one path in the subtree.");
        result.push_back(path_nodes->object);
        if (path_nodes->children.empty()) break;
        path_nodes = &(path_nodes->children[0]);
    }
    return result;
}

Path GeometryObject::Subtree::getLastPath() const {
    std::vector< shared_ptr<const GeometryObject> > result;
    if (empty()) return result;
    const GeometryObject::Subtree* path_nodes = this;
    while (true) {
        result.push_back(path_nodes->object);
        if (path_nodes->children.empty()) break;
        path_nodes = &(path_nodes->children.back());
    }
    return result;
}

void GeometryObject::ensureCanHasAsParent(const GeometryObject& potential_parent) const {
    if (isInSubtree(potential_parent))
        throw CyclicReferenceException();
}

void GeometryObject::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    //do nothing
}

std::size_t GeometryObject::getRealChildrenCount() const {
    return getChildrenCount();
}

shared_ptr<GeometryObject> GeometryObject::getRealChildAt(std::size_t child_nr) const {
    return getChildAt(child_nr);
}

void GeometryObject::removeAtUnsafe(std::size_t) {
    throw NotImplemented("removeAtUnsafe(std::size_t)");
}

void GeometryObject::forEachRealObjectInSubtree(std::function<bool (const GeometryObject &)> callback) const {
    if (!callback(*this)) return;
    std::size_t size = getRealChildrenCount();
    for (std::size_t i = 0; i < size; ++i) getRealChildAt(i)->forEachRealObjectInSubtree(callback);
}

// --- GeometryObjectD ---

template <int dims>
shared_ptr<const GeometryObject> GeometryObjectD<dims>::getMatchingAt(const DVec& point, const Predicate& predicate, const plask::PathHints* path) const {
    Subtree subtree = getPathsTo(point, false);
    // Walk the subtree
    const GeometryObject::Subtree* nodes = &subtree;
    while (!nodes->empty()) {
        if (predicate(*(nodes->object))) return nodes->object;
        if (nodes->children.empty()) return shared_ptr<const GeometryObject>();
        assert(nodes->children.size() == 1);
        if (path && nodes->object->isContainer()) {
            if (!path->include(nodes->object, nodes->children.front().object))
                return shared_ptr<const GeometryObject>();
        }
        nodes = &(nodes->children.front());
    }
    return shared_ptr<const GeometryObject>();
}

template <int dims>
std::set<std::string> GeometryObjectD<dims>::getClassesAt(const DVec& point, const plask::PathHints* path) const {
    std::set<std::string> result;
    getMatchingAt(point, [&](const GeometryObject& o) { result.insert(o.classes.begin(), o.classes.end()); return false; }, path);
    return result;
}

template struct GeometryObjectD<2>;
template struct GeometryObjectD<3>;


}   // namespace plask
