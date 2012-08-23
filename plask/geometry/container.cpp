#include "container.h"


namespace plask {

template <int dim>
void GeometryElementContainer<dim>::writeXMLChildAttr(XMLWriter::Element&, std::size_t, const AxisNames&) const {
    // do nothing
}

template <int dim>
void GeometryElementContainer<dim>::writeXML(XMLWriter::Element &parent_xml_element, const GeometryElement::WriteXMLCallback &write_cb, AxisNames axes) const {
    XMLWriter::Element container_tag = write_cb.makeTag(parent_xml_element, *this, axes);
    this->writeXMLAttr(container_tag, axes);
    for (std::size_t i = 0; i < children.size(); ++i) {
        XMLWriter::Element child_tag = write_cb.makeChildTag(container_tag, *this, i);
        writeXMLChildAttr(child_tag, i, axes);
        children[i]->getChild()->writeXML(child_tag, write_cb, axes);
    }
}

template <int dim>
typename GeometryElementContainer<dim>::Box GeometryElementContainer<dim>::getBoundingBox() const {
    if (children.empty()) return Box(Primitive<dim>::ZERO_VEC, Primitive<dim>::ZERO_VEC);
    Box result = children[0]->getBoundingBox();
    for (std::size_t i = 1; i < children.size(); ++i)
        result.makeInclude(children[i]->getBoundingBox());
    return result;
}

template <int dim>
shared_ptr<Material> GeometryElementContainer<dim>::getMaterial(const DVec& p) const {
    for (auto child_it = children.rbegin(); child_it != children.rend(); ++child_it) {
        shared_ptr<Material> r = (*child_it)->getMaterial(p);
        if (r != nullptr) return r;
    }
    return shared_ptr<Material>();
}

template <int dim>
void GeometryElementContainer<dim>::getBoundingBoxesToVec(const GeometryElement::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
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
void GeometryElementContainer<dim>::getElementsToVec(const GeometryElement::Predicate& predicate, std::vector< shared_ptr<const GeometryElement> >& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->shared_from_this());
        return;
    }
    if (path) {
        auto c = path->getTranslationChildren<dim>(*this);
        if (!c.empty()) {
            for (auto child: c) child->getElementsToVec(predicate, dest, path);
            return;
        }
    }
    for (auto child: children) child->getElementsToVec(predicate, dest, path);
}

template <int dim>
void GeometryElementContainer<dim>::getPositionsToVec(const GeometryElement::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
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

template <int dim>
bool GeometryElementContainer<dim>::isInSubtree(const GeometryElement& el) const {
    if (&el == this) return true;
    for (auto child: children)
        if (child->isInSubtree(el))
            return true;
    return false;
}

template <int dim>
GeometryElement::Subtree GeometryElementContainer<dim>::getPathsTo(const GeometryElement& el, const PathHints* path) const {
    if (this == &el) return this->shared_from_this();
    if (path) {
        auto hintChildren = path->getTranslationChildren<dim>(*this);
        if (!hintChildren.empty())
            return findPathsFromChildTo(hintChildren.begin(), hintChildren.end(), el, path);
    }
    return findPathsFromChildTo(children.begin(), children.end(), el, path);
}

template <int dim>
GeometryElement::Subtree GeometryElementContainer<dim>::getPathsTo(const GeometryElementContainer::DVec &point) const {
    GeometryElement::Subtree result;
    for (auto& child: children) {
        GeometryElement::Subtree child_path = child->getPathsTo(point);
        if (!child_path.empty())
            result.children.push_back(std::move(child_path));
    }
    if (!result.children.empty())
        result.element = this->shared_from_this();
    return result;
}

template <int dim>
bool GeometryElementContainer<dim>::removeIfTUnsafe(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
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
bool GeometryElementContainer<dim>::removeIfT(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
    if (removeIfTUnsafe(predicate)) {
        this->fireChildrenChanged();
        return true;
    } else
        return false;
}

template class GeometryElementContainer<2>;
template class GeometryElementContainer<3>;

template <>
void TranslationContainer<2>::writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const {
    shared_ptr<Translation<2>> child_tran = children[child_index];
    if (child_tran->translation.tran) dest_xml_child_tag.attr(axes.getNameForTran(), child_tran->translation.tran);
    if (child_tran->translation.up) dest_xml_child_tag.attr(axes.getNameForUp(), child_tran->translation.up);
}

template <>
void TranslationContainer<3>::writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const {
    shared_ptr<Translation<3>> child_tran = children[child_index];
    if (child_tran->translation.lon) dest_xml_child_tag.attr(axes.getNameForTran(), child_tran->translation.lon);
    if (child_tran->translation.tran) dest_xml_child_tag.attr(axes.getNameForTran(), child_tran->translation.tran);
    if (child_tran->translation.up) dest_xml_child_tag.attr(axes.getNameForUp(), child_tran->translation.up);
}

template class TranslationContainer<2>;
template class TranslationContainer<3>;


// ---- containers readers: ----

shared_ptr<GeometryElement> read_TranslationContainer2D(GeometryReader& reader) {
    shared_ptr< TranslationContainer<2> > result(new TranslationContainer<2>());
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    read_children<TranslationContainer<2>>(reader,
        [&]() {
            TranslationContainer<2>::DVec translation;
            translation.tran = reader.source.getAttribute(reader.getAxisTranName(), 0.0);
            translation.up = reader.source.getAttribute(reader.getAxisUpName(), 0.0);
            return result->add(reader.readExactlyOneChild< typename TranslationContainer<2>::ChildType >(), translation);
        },
        [&](const shared_ptr<typename TranslationContainer<2>::ChildType>& child) {
            result->add(child);
        }
    );
    return result;
}

shared_ptr<GeometryElement> read_TranslationContainer3D(GeometryReader& reader) {
    shared_ptr< TranslationContainer<3> > result(new TranslationContainer<3>());
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    read_children<TranslationContainer<3>>(reader,
        [&]() {
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



static GeometryReader::RegisterElementReader container2D_reader(TranslationContainer<2>::NAME, read_TranslationContainer2D);
static GeometryReader::RegisterElementReader container3D_reader(TranslationContainer<3>::NAME, read_TranslationContainer3D);

} // namespace plask
