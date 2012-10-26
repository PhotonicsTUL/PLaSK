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
    if (path) {
        auto c = path->getTranslationChildren<dim>(*this);
        if (!c.empty()) {
            for (auto child: c) child->getBoundingBoxesToVec(predicate, dest, path);
            return;
        }
    }
    for (auto child: children) child->getBoundingBoxesToVec(predicate, dest, path);
}

template <int dim>
void GeometryObjectContainer<dim>::getObjectsToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObject> >& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->shared_from_this());
        return;
    }
    if (path) {
        auto c = path->getTranslationChildren<dim>(*this);
        if (!c.empty()) {
            for (auto child: c) child->getObjectsToVec(predicate, dest, path);
            return;
        }
    }
    for (auto child: children) child->getObjectsToVec(predicate, dest, path);
}

template <int dim>
void GeometryObjectContainer<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    if (path) {
        auto c = path->getTranslationChildren<dim>(*this);
        if (!c.empty()) {
            for (auto child: c) child->getPositionsToVec(predicate, dest, path);
            return;
        }
    }
    for (auto child: children) child->getPositionsToVec(predicate, dest, path);
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
bool GeometryObjectContainer<dim>::isInSubtree(const GeometryObject& el) const {
    if (&el == this) return true;
    for (auto child: children)
        if (child->isInSubtree(el))
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
    for (auto child = children.rbegin(); child != children.rend(); ++child) {
        GeometryObject::Subtree child_path = (*child)->getPathsAt(point, all);
        if (!child_path.empty()) {
            result.children.push_back(std::move(child_path));
            if (!all) break;
        }
    }
    if (!result.children.empty())
        result.object = this->shared_from_this();
    return result;
}

template <int dim>
shared_ptr<const GeometryObject> GeometryObjectContainer<dim>::changedVersion(const GeometryObject::Changer& changer, Vec<3, double>* translation) const {
    shared_ptr<GeometryObject> result(const_pointer_cast<GeometryObject>(this->shared_from_this()));
    if (changer.apply(result, translation) || children.empty()) return result;

    bool were_changes = false;    //any children was changed?
    std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>> children_after_change;
    for (const shared_ptr<TranslationT>& child_tran: children) {
        //shared_ptr<const ChildType> new_child = child_tran->getChild();
        shared_ptr<GeometryObject> new_child = child_tran->getChild();
        Vec<3, double> trans_from_child;
        if (changer.apply(new_child, &trans_from_child)) were_changes = true;
        children_after_change.emplace_back(dynamic_pointer_cast<ChildType>(new_child), trans_from_child);
    }

    if (translation) *translation = vec(0.0, 0.0, 0.0); // we can't recommend nothing special
    if (were_changes) result = changedVersionForChildren(children_after_change, translation);

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

template struct GeometryObjectContainer<2>;
template struct GeometryObjectContainer<3>;

template <>
void TranslationContainer<2>::writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const {
    shared_ptr<Translation<2>> child_tran = children[child_index];
    if (child_tran->translation.tran() != 0.0) dest_xml_child_tag.attr(axes.getNameForTran(), child_tran->translation.tran());
    if (child_tran->translation.vert() != 0.0) dest_xml_child_tag.attr(axes.getNameForUp(), child_tran->translation.vert());
}

template <>
void TranslationContainer<3>::writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const {
    shared_ptr<Translation<3>> child_tran = children[child_index];
    if (child_tran->translation.lon() != 0.0) dest_xml_child_tag.attr(axes.getNameForTran(), child_tran->translation.lon());
    if (child_tran->translation.tran() != 0.0) dest_xml_child_tag.attr(axes.getNameForTran(), child_tran->translation.tran());
    if (child_tran->translation.vert() != 0.0) dest_xml_child_tag.attr(axes.getNameForUp(), child_tran->translation.vert());
}

template <int dim>
shared_ptr<GeometryObject> TranslationContainer<dim>::changedVersionForChildren(
        std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const {
    shared_ptr< TranslationContainer<dim> > result = make_shared< TranslationContainer<dim> >();
    for (std::size_t child_nr = 0; child_nr < children.size(); ++child_nr)
        if (children_after_change[child_nr].first)
            result->addUnsafe(children_after_change[child_nr].first, children[child_nr]->translation + vec<dim, double>(children_after_change[child_nr].second));
    return result;
}

template struct TranslationContainer<2>;
template struct TranslationContainer<3>;


// ---- containers readers: ----

shared_ptr<GeometryObject> read_TranslationContainer2D(GeometryReader& reader) {
    shared_ptr< TranslationContainer<2> > result(new TranslationContainer<2>());
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    read_children<TranslationContainer<2>>(reader,
        [&]() -> PathHints::Hint {
            TranslationContainer<2>::DVec translation;
            translation.tran() = reader.source.getAttribute(reader.getAxisTranName(), 0.0);
            translation.vert() = reader.source.getAttribute(reader.getAxisUpName(), 0.0);
            return result->add(reader.readExactlyOneChild< typename TranslationContainer<2>::ChildType >(), translation);
        },
        [&](const shared_ptr<typename TranslationContainer<2>::ChildType>& child) {
            result->add(child);
        }
    );
    return result;
}

shared_ptr<GeometryObject> read_TranslationContainer3D(GeometryReader& reader) {
    shared_ptr< TranslationContainer<3> > result(new TranslationContainer<3>());
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    read_children<TranslationContainer<3>>(reader,
        [&]() -> PathHints::Hint {
            TranslationContainer<3>::DVec translation;
            translation.c0 = reader.source.getAttribute(reader.getAxisName(0), 0.0);
            translation.c1 = reader.source.getAttribute(reader.getAxisName(1), 0.0);
            translation.c2 = reader.source.getAttribute(reader.getAxisName(2), 0.0);
            return result->add(reader.readExactlyOneChild< typename TranslationContainer<3>::ChildType >(), translation);
        },
        [&](const shared_ptr<typename TranslationContainer<3>::ChildType>& child) {
            result->add(child);
        }
    );
    return result;
}



static GeometryReader::RegisterObjectReader container2D_reader(TranslationContainer<2>::NAME, read_TranslationContainer2D);
static GeometryReader::RegisterObjectReader container3D_reader(TranslationContainer<3>::NAME, read_TranslationContainer3D);

} // namespace plask
