#include "container.h"

#include "manager.h"


namespace plask {

template <int dim>
typename GeometryElementContainer<dim>::Box GeometryElementContainer<dim>::getBoundingBox() const {
    if (children.empty()) return Box(Primitive<dim>::ZERO_VEC, Primitive<dim>::ZERO_VEC);
    Box result = children[0]->getBoundingBox();
    for (std::size_t i = 1; i < children.size(); ++i)
        result.include(children[i]->getBoundingBox());
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
        dest.push_back(getBoundingBox());
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
GeometryElement::Subtree GeometryElementContainer<dim>::findPathsTo(const GeometryElement& el, const PathHints* path) const {
    if (this == &el) return this->shared_from_this();
    if (path) {
        auto hintChildren = path->getTranslationChildren<dim>(*this);
        if (!hintChildren.empty())
            return findPathsFromChildTo(hintChildren.begin(), hintChildren.end(), el, path);
    }
    return findPathsFromChildTo(children.begin(), children.end(), el, path);
}

template <int dim>
bool GeometryElementContainer<dim>::childrenEraseFromEnd(typename TranslationVector::iterator firstToErase) {
    if (firstToErase != children.end()) {
        children.erase(firstToErase, children.end());
        fireChildrenChanged();
        return true;
    } else
        return false;
}

template <int dim>
bool GeometryElementContainer<dim>::removeT(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
    auto dst = children.begin();
    for (auto i: children)
        if (predicate(i))
            disconnectOnChildChanged(*i);
        else
            *dst++ = i;
    return childrenEraseFromEnd(dst);
}

template class GeometryElementContainer<2>;
template class GeometryElementContainer<3>;

// ---- containers readers: ----

shared_ptr<GeometryElement> read_TranslationContainer2d(GeometryReader& reader) {
    shared_ptr< TranslationContainer<2> > result(new TranslationContainer<2>());
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    read_children<TranslationContainer<2>>(reader,
        [&]() {
            TranslationContainer<2>::DVec translation;
            translation.tran = reader.source.getAttribute(reader.getAxisLonName(), 0.0);
            translation.up = reader.source.getAttribute(reader.getAxisUpName(), 0.0);
            return result->add(reader.readExactlyOneChild< typename TranslationContainer<2>::ChildType >(), translation);
        },
        [&](const shared_ptr<typename TranslationContainer<2>::ChildType>& child) {
            result->add(child);
        }
    );
    return result;
}

shared_ptr<GeometryElement> read_TranslationContainer3d(GeometryReader& reader) {
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



static GeometryReader::RegisterElementReader container2d_reader("container" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D, read_TranslationContainer2d);
static GeometryReader::RegisterElementReader container3d_reader("container" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D, read_TranslationContainer3d);


} // namespace plask
