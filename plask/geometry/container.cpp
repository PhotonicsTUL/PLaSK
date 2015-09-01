#include "container.h"


namespace plask {

template <int dim>
void GeometryObjectContainer<dim>::writeXMLChildAttr(XMLWriter::Element&, std::size_t, const AxisNames&) const {
    // do nothing
}

template <int dim>
void GeometryObjectContainer<dim>::writeXML(XMLWriter::Element &parent_xml_object, GeometryObject::WriteXMLCallback &write_cb, AxisNames axes) const {
    XMLWriter::Element container_tag = write_cb.makeTag(parent_xml_object, *this, axes);
    if (GeometryObject::WriteXMLCallback::isRef(container_tag)) return;
    this->writeXMLAttr(container_tag, axes);
    for (std::size_t i = 0; i < children.size(); ++i) {
        XMLWriter::Element child_tag = write_cb.makeChildTag(container_tag, *this, i);
        writeXMLChildAttr(child_tag, i, axes);
        children[i]->getChild()->writeXML(child_tag, write_cb, axes);
    }
}

template <int dim>
void GeometryObjectContainer<dim>::onChildChanged(const GeometryObject::Event &evt) {
    this->fireChanged(evt.oryginalSource(), evt.flagsForParent());
}

template <int dim>
void GeometryObjectContainer<dim>::connectOnChildChanged(Translation<dim> &child) {
    child.changedConnectMethod(this, &GeometryObjectContainer::onChildChanged);
}

template <int dim>
void GeometryObjectContainer<dim>::disconnectOnChildChanged(Translation<dim> &child) {
    child.changedDisconnectMethod(this, &GeometryObjectContainer::onChildChanged);
}

template <int dim>
bool GeometryObjectContainer<dim>::contains(const GeometryObjectContainer::DVec &p) const {
    for (auto child: children) if (child->contains(p)) return true;
    return false;
}

template <int dim>
typename GeometryObjectContainer<dim>::Box GeometryObjectContainer<dim>::getBoundingBox() const {
    if (children.empty()) return Box(Primitive<dim>::ZERO_VEC, Primitive<dim>::ZERO_VEC);
    Box result = children[0]->getBoundingBox();
    for (std::size_t i = 1; i < children.size(); ++i)
        result.makeInclude(children[i]->getBoundingBox());
    return result;
}

template <int dim>
shared_ptr<Material> GeometryObjectContainer<dim>::getMaterial(const DVec& p) const {
    for (auto child_it = children.rbegin(); child_it != children.rend(); ++child_it) {
        shared_ptr<Material> r = (*child_it)->getMaterial(p);
        if (r != nullptr) return r;
    }
    return shared_ptr<Material>();
}

template <int dim>
void GeometryObjectContainer<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->getBoundingBox());
        return;
    }
    forEachChild([&](const Translation<dim> &child) { child.getBoundingBoxesToVec(predicate, dest, path); }, path);

    /*if (path) {
        auto c = path->getTranslationChildren<dim>(*this);
        if (!c.empty()) {
            for (auto child: c) child->getBoundingBoxesToVec(predicate, dest, path);
            return;
        }
    }
    for (auto child: children) child->getBoundingBoxesToVec(predicate, dest, path);*/
}

template <int dim>
void GeometryObjectContainer<dim>::getObjectsToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObject> >& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->shared_from_this());
        return;
    }
    forEachChild([&](const Translation<dim> &child) { child.getObjectsToVec(predicate, dest, path); }, path);

    /*if (path) {
        auto c = path->getTranslationChildren<dim>(*this);
        if (!c.empty()) {
            for (auto child: c) child->getObjectsToVec(predicate, dest, path);
            return;
        }
    }
    for (auto child: children) child->getObjectsToVec(predicate, dest, path);*/
}

template <int dim>
void GeometryObjectContainer<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    forEachChild([&](const Translation<dim> &child) { child.getPositionsToVec(predicate, dest, path); }, path);

    /*if (path) {
        auto c = path->getTranslationChildren<dim>(*this);
        if (!c.empty()) {
            for (auto child: c) child->getPositionsToVec(predicate, dest, path);
            return;
        }
    }
    for (auto child: children) child->getPositionsToVec(predicate, dest, path);*/
}

// template <int dim>
// void GeometryObjectContainer<dim>::extractToVec(const GeometryObject::Predicate &predicate, std::vector< shared_ptr<const GeometryObjectD<dim> > >& dest, const PathHints *path) const {
//     if (predicate(*this)) {
//         dest.push_back(static_pointer_cast< const GeometryObjectD<dim> >(this->shared_from_this()));
//         return;
//     }
//     if (path) {
//         auto c = path->getTranslationChildren<dim>(*this);
//         if (!c.empty()) {
//             for (auto child: c) child->extractToVec(predicate, dest, path);
//             return;
//         }
//     }
//     for (auto child: children) child->extractToVec(predicate, dest, path);
// }

template <int dim>
bool GeometryObjectContainer<dim>::hasInSubtree(const GeometryObject& el) const {
    if (&el == this) return true;
    for (auto child: children)
        if (child->hasInSubtree(el))
            return true;
    return false;
}

template <int dim>
GeometryObject::Subtree GeometryObjectContainer<dim>::getPathsTo(const GeometryObject& el, const PathHints* path) const {
    if (this == &el) return this->shared_from_this();
    if (path) {
        auto hintChildren = path->getTranslationChildren<dim>(*this);
        if (!hintChildren.empty())
            return findPathsFromChildTo(hintChildren.begin(), hintChildren.end(), el, path);
    }
    return findPathsFromChildTo(children.begin(), children.end(), el, path);
}

template <int dim>
GeometryObject::Subtree GeometryObjectContainer<dim>::getPathsAt(const GeometryObjectContainer::DVec &point, bool all) const {
    GeometryObject::Subtree result;
    if (all) {
        for (auto child = children.begin(); child != children.end(); ++child) {
            GeometryObject::Subtree child_path = (*child)->getPathsAt(point, true);
            if (!child_path.empty())
                result.children.push_back(std::move(child_path));
        }
    } else {
        for (auto child = children.rbegin(); child != children.rend(); ++child) {
            GeometryObject::Subtree child_path = (*child)->getPathsAt(point, false);
            if (!child_path.empty()) {
                result.children.push_back(std::move(child_path));
                break;
            }
        }
    }
    if (!result.children.empty())
        result.object = this->shared_from_this();
    return result;
}

template <int dim>
std::size_t GeometryObjectContainer<dim>::getChildrenCount() const { return children.size(); }

template <int dim>
shared_ptr<GeometryObject> GeometryObjectContainer<dim>::getChildNo(std::size_t child_no) const {
    this->ensureIsValidChildNr(child_no);
    return children[child_no];
}

template <int dim>
shared_ptr<const GeometryObject> GeometryObjectContainer<dim>::changedVersion(const GeometryObject::Changer& changer, Vec<3, double>* translation) const {
    shared_ptr<GeometryObject> result(const_pointer_cast<GeometryObject>(this->shared_from_this()));
    if (changer.apply(result, translation) || children.empty()) return result;

    bool were_changes = false;    //any children was changed?
    std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>> children_after_change;
    for (const shared_ptr<TranslationT>& child_tran: children) {
        Vec<3, double> trans_from_child;
        shared_ptr<GeometryObject> old_child = child_tran->getChild();
        shared_ptr<GeometryObject> new_child = const_pointer_cast<GeometryObject>(old_child->changedVersion(changer, &trans_from_child));
        if (new_child != old_child) were_changes = true;
        children_after_change.emplace_back(dynamic_pointer_cast<ChildType>(new_child), trans_from_child);
    }

    if (translation) *translation = vec(0.0, 0.0, 0.0); // we can't recommend nothing special
    if (were_changes) result = this->changedVersionForChildren(children_after_change, translation);

    return result;
}

template <int dim>
bool GeometryObjectContainer<dim>::removeIfTUnsafe(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
    auto dst = children.begin();
    for (auto i: children)
        if (predicate(i))
            disconnectOnChildChanged(*i);
        else
            *dst++ = i;
    if (dst != children.end()) {
        children.erase(dst, children.end());
        return true;
    } else
        return false;
}

template <int dim>
bool GeometryObjectContainer<dim>::removeIfT(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
    if (removeIfTUnsafe(predicate)) {
        this->fireChildrenChanged();
        return true;
    } else
        return false;
}

template <int dim>
void GeometryObjectContainer<dim>::removeAtUnsafe(std::size_t index) {
    disconnectOnChildChanged(*children[index]);
    children.erase(children.begin() + index);
}

template struct PLASK_API GeometryObjectContainer<2>;
template struct PLASK_API GeometryObjectContainer<3>;


} // namespace plask
