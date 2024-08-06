/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "stack.hpp"
#include "separator.hpp"

#define PLASK_STACK2D_NAME ("stack" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D)
#define PLASK_STACK3D_NAME ("stack" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D)
#define PLASK_SHELF_NAME "shelf"

#define BASEH_ATTR "shift"
#define REPEAT_ATTR "repeat"
#define REQUIRE_EQUAL_HEIGHTS_ATTR "flat"
#define ZERO_ATTR "zero"

namespace plask {

template <int dim, typename Primitive<dim>::Direction growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::setBaseHeight(double newBaseHeight) {
    if (getBaseHeight() == newBaseHeight) return;
    double diff = newBaseHeight - getBaseHeight();
    stackHeights.front() = newBaseHeight;
    for (std::size_t i = 1; i < stackHeights.size(); ++i) {
        stackHeights[i] += diff;
        children[i - 1]->translation[growingDirection] += diff;
        // children[i-1]->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }
    this->fireChildrenChanged();  // TODO should this be called? children was moved but not removed/inserted
    // this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    // TODO: block connection to not react on children changed, call
    // children[i]->fireChanged(GeometryObject::Event::EVENT_RESIZE); for each child, call
    // this->fireChanged(GeometryObject::Event::EVENT_RESIZE delegate;
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::setZeroBefore(std::size_t index) {
    std::size_t h_count = stackHeights.size();
    if (index >= h_count) throw OutOfBoundsException("setZeroBefore", "index", index, 0, h_count - 1);
    setBaseHeight(stackHeights[0] - stackHeights[index]);
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::alignZeroOn(std::size_t index, double pos) {
    std::size_t h_count = children.size();
    if (index >= h_count) throw OutOfBoundsException("alignZeroOn", "index", index, 0, h_count - 1);
    auto child = children[index]->getChild();
    double shift = child ? child->getBoundingBox().lower[growingDirection] : 0.;
    setBaseHeight(stackHeights[0] - stackHeights[index] + shift - pos);
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
std::size_t StackContainerBaseImpl<dim, growingDirection>::getInsertionIndexForHeight(double height) const {
    return std::lower_bound(stackHeights.begin(), stackHeights.end(), height) - stackHeights.begin();
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
const shared_ptr<typename StackContainerBaseImpl<dim, growingDirection>::TranslationT>
StackContainerBaseImpl<dim, growingDirection>::getChildForHeight(
    double height,
    shared_ptr<typename StackContainerBaseImpl<dim, growingDirection>::TranslationT>& sec_candidate) const {
    auto it = std::lower_bound(stackHeights.begin(), stackHeights.end(), height);
    // we have: *it >= height > *(it-1)
    if (it == stackHeights.end()) {  // height > stackHeights.back()
        if (is_zero(height - stackHeights.back(), 16 * SMALL) && !children.empty()) return children.back();
        return shared_ptr<TranslationT>();
    }
    if (it == stackHeights.begin()) {  // stackHeights.front() >= height
        if (is_zero(stackHeights.front() - height,
                    16 * SMALL))  // children.empty() now impossible - then it == stackHeights.end()
            return children[0];
        else
            return shared_ptr<TranslationT>();
    }
    std::ptrdiff_t sh_index = it - stackHeights.begin();
    if (sh_index > 1 && is_zero(height - stackHeights[sh_index-1], 16 * SMALL))
        sec_candidate = children[sh_index-2];
    else if (sh_index + 1 < stackHeights.size() && is_zero(stackHeights[sh_index] - height, 16 * SMALL))
        sec_candidate = children[sh_index];
    return children[sh_index-1];
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
bool StackContainerBaseImpl<dim, growingDirection>::contains(
    const typename StackContainerBaseImpl<dim, growingDirection>::DVec& p) const {
    shared_ptr<TranslationT> sec_candidate;
    const shared_ptr<TranslationT> c = getChildForHeight(p[growingDirection], sec_candidate);
    if (c) {
        if (c->contains(p)) return true;
        if (sec_candidate) return sec_candidate->contains(p);
    }
    return false;
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
shared_ptr<Material> StackContainerBaseImpl<dim, growingDirection>::getMaterial(
    const typename StackContainerBaseImpl<dim, growingDirection>::DVec& p) const {
    shared_ptr<TranslationT> sec_candidate;
    const shared_ptr<TranslationT> c = getChildForHeight(p[growingDirection], sec_candidate);
    if (c) {
        if (auto material = c->getMaterial(p)) return material;
        if (sec_candidate) return sec_candidate->getMaterial(p);
    }
    return shared_ptr<Material>();
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
GeometryObject::Subtree StackContainerBaseImpl<dim, growingDirection>::getPathsAt(
    const typename StackContainerBaseImpl<dim, growingDirection>::DVec& point,
    bool all) const {
    shared_ptr<TranslationT> sec_candidate;
    const shared_ptr<TranslationT> c = getChildForHeight(point[growingDirection], sec_candidate);
    GeometryObject::Subtree result;
    if (c) {
        GeometryObject::Subtree child_path = c->getPathsAt(point, all);
        if (!child_path.empty()) {
            result.children.push_back(std::move(child_path));
            if (!all) {  // one is enough
                result.object = this->shared_from_this();
                return result;
            }
        }
        if (sec_candidate) {
            child_path = sec_candidate->getPathsAt(point, all);
            if (!child_path.empty()) result.children.push_back(std::move(child_path));
        }
        if (!result.children.empty())  // c or sec_candidate or both have given us some children
            result.object = this->shared_from_this();
    }
    return result;
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::removeAtUnsafe(std::size_t index) {
    GeometryObjectContainer<dim>::removeAtUnsafe(index);
    stackHeights.pop_back();
    updateAllHeights(index);
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::onChildChanged(const GeometryObject::Event& evt) {
    if (evt.isResize())
        updateAllHeights();  // TODO optimization: find evt source index and update size from this index to back
    this->fireChanged(evt.originalSource(), evt.flagsForParent());
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::updateAllHeights(std::size_t first_child_index) {
    for (; first_child_index < children.size(); ++first_child_index) updateHeight(first_child_index);

    updateAllHeights();  //! to use AccurateSum
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::updateAllHeights() {
    AccurateSum sum = stackHeights[0];
    for (std::size_t child_index = 0; child_index < children.size(); ++child_index) {
        auto child = children[child_index]->getChild();
        auto elBoudingBox = child ? child->getBoundingBox() : Box(Primitive<dim>::ZERO_VEC, Primitive<dim>::ZERO_VEC);
        sum -= elBoudingBox.lower[growingDirection];
        children[child_index]->translation[growingDirection] = sum;
        sum += elBoudingBox.upper[growingDirection];
        stackHeights[child_index + 1] = sum;
    }
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::rebuildStackHeights(std::size_t first_child_index) {
    stackHeights.resize(children.size() + 1);
    updateAllHeights(first_child_index);
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::writeXMLAttr(XMLWriter::Element& dest_xml_object,
                                                                 const AxisNames& axes) const {
    BaseClass::writeXMLAttr(dest_xml_object, axes);
    dest_xml_object.attr(BASEH_ATTR, getBaseHeight());
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
bool StackContainerBaseImpl<dim, growingDirection>::removeIfTUnsafe(
    const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
    if (GeometryObjectContainer<dim>::removeIfTUnsafe(predicate)) {
        this->rebuildStackHeights();
        return true;
    } else
        return false;
}

template struct PLASK_API StackContainerBaseImpl<2, Primitive<2>::DIRECTION_VERT>;
template struct PLASK_API StackContainerBaseImpl<3, Primitive<3>::DIRECTION_VERT>;
template struct PLASK_API StackContainerBaseImpl<2, Primitive<2>::DIRECTION_TRAN>;

/*template <int dim>    //this is fine but GeometryObjects doesn't have copy constructors at all, because signal doesn't
have copy constructor StackContainer<dim>::StackContainer(const StackContainer& to_copy) :
StackContainerBaseImpl<dim>(to_copy) //copy all but aligners
{
    std::vector<Aligner*> aligners_copy;
    aligners_copy.reserve(to_copy.size());
    for (auto a: to_copy.aligners) aligners_copy.push_back(a->clone());
    this->aligners = aligners_copy;
}*/

template <> const StackContainer<2>::ChildAligner& StackContainer<2>::DefaultAligner() {
    static auto leftZeroAl = align::left(0);
    return leftZeroAl;
}

template <> const StackContainer<3>::ChildAligner& StackContainer<3>::DefaultAligner() {
    static auto leftBackAl = align::left(0) & align::back(0);
    return leftBackAl;
}

template <int dim>
PathHints::Hint StackContainer<dim>::insertUnsafe(const shared_ptr<ChildType>& el,
                                                  const std::size_t pos,
                                                  const ChildAligner& aligner) {
    const auto bb = el ? el->getBoundingBox() : Box(Primitive<dim>::ZERO_VEC, Primitive<dim>::ZERO_VEC);
    shared_ptr<TranslationT> trans_geom = newTranslation(el, aligner, stackHeights[pos] - bb.lower.vert(), bb);
    this->connectOnChildChanged(*trans_geom);
    children.insert(children.begin() + pos, trans_geom);
    this->aligners.insert(this->aligners.begin() + pos, aligner);
    stackHeights.insert(stackHeights.begin() + pos, stackHeights[pos]);
    const double delta = bb.upper.vert() - bb.lower.vert();
    for (std::size_t i = pos + 1; i < children.size(); ++i) {
        stackHeights[i] += delta;
        children[i]->translation.vert() += delta;
    }
    stackHeights.back() += delta;

    this->updateAllHeights();  //! to use AccurateSum

    this->fireChildrenInserted(pos, pos + 1);
    return PathHints::Hint(shared_from_this(), trans_geom);
}

template <int dim>
shared_ptr<typename StackContainer<dim>::TranslationT> StackContainer<dim>::newTranslation(
    const shared_ptr<typename StackContainer<dim>::ChildType>& el,
    const typename StackContainer<dim>::ChildAligner& aligner,
    double up_trans,
    const typename StackContainer<dim>::Box& elBB) const {
    shared_ptr<TranslationT> result(new TranslationT(el, Primitive<dim>::ZERO_VEC));
    result->translation.vert() = up_trans;
    aligner.align(*result, elBB);
    // el->fireChanged();    //??
    return result;
}

template <int dim>
shared_ptr<typename StackContainer<dim>::TranslationT> StackContainer<dim>::newTranslation(
    const shared_ptr<typename StackContainer<dim>::ChildType>& el,
    const typename StackContainer<dim>::ChildAligner& aligner,
    double up_trans) const {
    shared_ptr<TranslationT> result(new TranslationT(el, Primitive<dim>::ZERO_VEC));
    result->translation.vert() = up_trans;
    aligner.align(*result);
    return result;
}

template <int dim> void StackContainer<dim>::onChildChanged(const GeometryObject::Event& evt) {
    if (evt.isResize()) {
        ParentClass::align(const_cast<TranslationT*>(evt.source<TranslationT>()));
        this->updateAllHeights();  // TODO optimization: find evt source index and update size from this index to back
    }
    this->fireChanged(evt.originalSource(), evt.flagsForParent());
}

template <int dim>
PathHints::Hint StackContainer<dim>::addUnsafe(const shared_ptr<typename StackContainer<dim>::ChildType>& el,
                                               const typename StackContainer<dim>::ChildAligner& aligner) {
    double el_translation, next_height;
    auto elBB = el ? el->getBoundingBox() : Box(Primitive<dim>::ZERO_VEC, Primitive<dim>::ZERO_VEC);
    this->calcHeight(elBB, stackHeights.back(), el_translation, next_height);
    shared_ptr<TranslationT> trans_geom = newTranslation(el, aligner, el_translation, elBB);
    this->connectOnChildChanged(*trans_geom);
    children.push_back(trans_geom);
    stackHeights.push_back(next_height);
    this->aligners.push_back(aligner);

    this->updateAllHeights();  //! to use AccurateSum

    this->fireChildrenInserted(children.size() - 1, children.size());
    return PathHints::Hint(shared_from_this(), trans_geom);
}

template <int dim>
bool StackContainer<dim>::removeIfTUnsafe(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
    if (ParentClass::removeIfTUnsafe(predicate)) {
        this->rebuildStackHeights();
        return true;
    } else
        return false;
    /*    auto dst = children.begin();
    auto al_dst = this->aligners.begin();
    auto al_src = this->aligners.begin();
    for (auto i: children) {
        if (predicate(i))
            this->disconnectOnChildChanged(*i);
        else {
            *dst++ = i;
            *al_dst++ = std::move(*al_src);
        }
        ++al_src;
    }
    if (dst != children.end()) {
        children.erase(dst, children.end());
        this->aligners.erase(al_dst, this->aligners.end());
        this->rebuildStackHeights();
        return true;
    } else
        return false;*/
}

template <int dim> void StackContainer<dim>::removeAtUnsafe(std::size_t index) {
    GeometryObjectContainer<dim>::removeAtUnsafe(index);
    this->aligners.erase(this->aligners.begin() + index);
    stackHeights.pop_back();
    this->updateAllHeights(index);
}

template <int dim>
void StackContainer<dim>::writeXML(XMLWriter::Element& parent_xml_object,
                                   GeometryObject::WriteXMLCallback& write_cb,
                                   AxisNames axes) const {
    XMLWriter::Element container_tag = write_cb.makeTag(parent_xml_object, *this, axes);
    if (GeometryObject::WriteXMLCallback::isRef(container_tag)) return;
    this->writeXMLAttr(container_tag, axes);
    for (int i = int(children.size()) - 1; i >= 0; --i) {  // children are written in reverse order
        XMLWriter::Element child_tag = write_cb.makeChildTag(container_tag, *this, i);
        writeXMLChildAttr(child_tag, i, axes);
        if (auto child = children[i]->getChild()) child->writeXML(child_tag, write_cb, axes);
    }
}

template <int dim>
void StackContainer<dim>::writeXMLChildAttr(XMLWriter::Element& dest_xml_child_tag,
                                            std::size_t child_index,
                                            const AxisNames& axes) const {
    this->aligners[child_index].writeToXML(dest_xml_child_tag, axes);
}

template <int dim> shared_ptr<GeometryObject> StackContainer<dim>::shallowCopy() const {
    shared_ptr<StackContainer<dim>> result = plask::make_shared<StackContainer<dim>>(this->getBaseHeight());
    result->default_aligner = default_aligner;
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        result->addUnsafe(this->children[child_no]->getChild(), this->aligners[child_no]);
    return result;
}

template <int dim>
shared_ptr<GeometryObject> StackContainer<dim>::deepCopy(
    std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const {
    auto found = copied.find(this);
    if (found != copied.end()) return found->second;
    shared_ptr<StackContainer<dim>> result = plask::make_shared<StackContainer<dim>>(this->getBaseHeight());
    result->default_aligner = default_aligner;
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (this->children[child_no]->getChild())
            result->addUnsafe(static_pointer_cast<ChildType>(this->children[child_no]->getChild()->deepCopy(copied)),
                              this->aligners[child_no]);
    copied[this] = result;
    return result;
}

template <int dim>
shared_ptr<GeometryObject> StackContainer<dim>::changedVersionForChildren(
    std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change,
    Vec<3, double>* /*recomended_translation*/) const {
    shared_ptr<StackContainer<dim>> result = plask::make_shared<StackContainer<dim>>(this->getBaseHeight());
    result->default_aligner = default_aligner;
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (children_after_change[child_no].first)
            result->addUnsafe(children_after_change[child_no].first, this->aligners[child_no]);
    return result;
}

template <int dim> const char* StackContainer<dim>::NAME = dim == 2 ? PLASK_STACK2D_NAME : PLASK_STACK3D_NAME;

template struct PLASK_API StackContainer<2>;
template struct PLASK_API StackContainer<3>;

const char* ShelfContainer2D::NAME = PLASK_SHELF_NAME;

PathHints::Hint ShelfContainer2D::addGap(double size) {
    return addUnsafe(plask::make_shared<Gap1D<2, Primitive<2>::DIRECTION_TRAN>>(size));
}

bool ShelfContainer2D::isFlat() const {
    std::size_t first = 0;
    while (first < children.size() && children[first]->childHasType(GeometryObject::TYPE_SEPARATOR)) ++first;
    if (first + 2 >= children.size())  // will we outside when we go 2 steps forward?
        return true;                   //(almost) same separators
    const double height = children[first]->getBoundingBoxSize().vert();
    for (std::size_t i = first + 1; i < children.size(); ++i)
        if (!children[i]->childHasType(GeometryObject::TYPE_SEPARATOR) &&
            !is_zero(height - children[i]->getBoundingBoxSize().vert()))
            return false;
    return true;
}

PathHints::Hint ShelfContainer2D::addUnsafe(const shared_ptr<ChildType>& el) {
    if (!el) return PathHints::Hint(shared_from_this(), shared_ptr<GeometryObject>());

    double el_translation, next_height;
    auto elBB = el->getBoundingBox();
    calcHeight(elBB, stackHeights.back(), el_translation, next_height);
    shared_ptr<TranslationT> trans_geom = plask::make_shared<TranslationT>(el, vec(el_translation, -elBB.lower.vert()));
    connectOnChildChanged(*trans_geom);
    children.push_back(trans_geom);
    stackHeights.push_back(next_height);

    this->updateAllHeights();  //! to use AccurateSum

    this->fireChildrenInserted(children.size() - 1, children.size());
    return PathHints::Hint(shared_from_this(), trans_geom);
}

PathHints::Hint ShelfContainer2D::insertUnsafe(const shared_ptr<ChildType>& el, const std::size_t pos) {
    if (!el) return PathHints::Hint(shared_from_this(), shared_ptr<GeometryObject>());

    const auto bb = el->getBoundingBox();
    shared_ptr<TranslationT> trans_geom =
        plask::make_shared<TranslationT>(el, vec(stackHeights[pos] - bb.lower.tran(), -bb.lower.vert()));
    connectOnChildChanged(*trans_geom);
    children.insert(children.begin() + pos, trans_geom);
    stackHeights.insert(stackHeights.begin() + pos, stackHeights[pos]);
    const double delta = bb.upper.tran() - bb.lower.tran();
    for (std::size_t i = pos + 1; i < children.size(); ++i) {
        stackHeights[i] += delta;
        children[i]->translation.tran() += delta;
    }
    stackHeights.back() += delta;

    this->updateAllHeights();  //! to use AccurateSum

    this->fireChildrenInserted(pos, pos + 1);
    return PathHints::Hint(shared_from_this(), trans_geom);
}

shared_ptr<GeometryObject> ShelfContainer2D::shallowCopy() const {
    shared_ptr<ShelfContainer2D> result = plask::make_shared<ShelfContainer2D>(this->getBaseHeight());
    result->resizableGap = resizableGap;
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        result->addUnsafe(this->children[child_no]->getChild());
    return result;
}

shared_ptr<GeometryObject> ShelfContainer2D::deepCopy(
    std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const {
    auto found = copied.find(this);
    if (found != copied.end()) return found->second;
    shared_ptr<ShelfContainer2D> result = plask::make_shared<ShelfContainer2D>(this->getBaseHeight());
    result->resizableGap = resizableGap;
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (this->children[child_no]->getChild())
            result->addUnsafe(static_pointer_cast<ChildType>(this->children[child_no]->getChild()->deepCopy(copied)));
    copied[this] = result;
    return result;
}

shared_ptr<GeometryObject> ShelfContainer2D::changedVersionForChildren(
    std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change,
    Vec<3, double>* /*recomended_translation*/) const {
    shared_ptr<ShelfContainer2D> result = plask::make_shared<ShelfContainer2D>(this->getBaseHeight());
    result->resizableGap = resizableGap;
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (children_after_change[child_no].first) result->addUnsafe(children_after_change[child_no].first);
    return result;
}

void ShelfContainer2D::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    BaseClass::writeXMLAttr(dest_xml_object, axes);
    dest_xml_object.attr(REQUIRE_EQUAL_HEIGHTS_ATTR, false);
}

template <typename UpperClass> bool MultiStackContainer<UpperClass>::reduceHeight(double& height) const {
    const double zeroBasedStackHeight = stackHeights.back() - stackHeights.front();
    const double zeroBasedRequestHeight = height - stackHeights.front();
    if (zeroBasedRequestHeight < 0.0 || zeroBasedRequestHeight > zeroBasedStackHeight * repeat_count) return false;
    height = std::fmod(zeroBasedRequestHeight, zeroBasedStackHeight) + stackHeights.front();
    return true;
}

template <typename UpperClass>
typename MultiStackContainer<UpperClass>::Box MultiStackContainer<UpperClass>::getBoundingBox() const {
    Box result = UpperClass::getBoundingBox();
    result.upper[UpperClass::GROWING_DIR] =
        result.lower[UpperClass::GROWING_DIR] +
        (result.upper[UpperClass::GROWING_DIR] - result.lower[UpperClass::GROWING_DIR]) * repeat_count;
    return result;
}

template <typename UpperClass>
typename MultiStackContainer<UpperClass>::Box MultiStackContainer<UpperClass>::getRealBoundingBox() const {
    return UpperClass::getBoundingBox();
}

// TODO good but unused
/*template <typename UpperClass>
bool MultiStackContainer<UpperClass>::intersects(const Box& area) const {
    const double minusZeroBasedStackHeight = stackHeights.front() - stackHeights.back();
    for (unsigned r = 0; r < repeat_count; ++r)
        if (UpperClass::intersects(area.translatedUp(minusZeroBasedStackHeight*r)))
            return true;
    return false;
}*/

template <typename UpperClass>
void MultiStackContainer<UpperClass>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate,
                                                            std::vector<Box>& dest,
                                                            const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    if (repeat_count == 0) return;
    std::size_t old_size = dest.size();
    UpperClass::getBoundingBoxesToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    const double stackHeight = stackHeights.back() - stackHeights.front();
    for (unsigned r = 1; r < repeat_count; ++r) {
        for (std::size_t i = old_size; i < new_size; ++i) dest.push_back(dest[i]);
        for (auto i = dest.end() - (new_size - old_size); i != dest.end(); ++i)
            i->translateDir(UpperClass::GROWING_DIR, stackHeight * r);
    }
}

template <typename UpperClass>
void MultiStackContainer<UpperClass>::getObjectsToVec(const GeometryObject::Predicate& predicate,
                                                      std::vector<shared_ptr<const GeometryObject>>& dest,
                                                      const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->shared_from_this());
        return;
    }
    if (repeat_count == 0) return;
    std::size_t old_size = dest.size();
    UpperClass::getObjectsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (unsigned r = 1; r < repeat_count; ++r)
        for (std::size_t i = old_size; i < new_size; ++i) dest.push_back(dest[i]);
}

template <typename UpperClass>
void MultiStackContainer<UpperClass>::getPositionsToVec(const GeometryObject::Predicate& predicate,
                                                        std::vector<DVec>& dest,
                                                        const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<MultiStackContainer<UpperClass>::DIM>::ZERO_VEC);
        return;
    }
    if (repeat_count == 0) return;
    std::size_t old_size = dest.size();
    UpperClass::getPositionsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    const double stackHeight = stackHeights.back() - stackHeights.front();
    for (unsigned r = 1; r < repeat_count; ++r)
        for (std::size_t i = old_size; i < new_size; ++i) {
            dest.push_back(dest[i]);
            dest.back()[UpperClass::GROWING_DIR] += stackHeight * r;
        }
}

// template <typename UpperClass>
// void MultiStackContainer<UpperClass>::extractToVec(const GeometryObject::Predicate &predicate, std::vector<
// shared_ptr<const GeometryObjectD<dim> > >& dest, const PathHints *path) const {
//     if (predicate(*this)) {
//         dest.push_back(static_pointer_cast< const GeometryObjectD<dim> >(this->shared_from_this()));
//         return;
//     }
//     std::size_t old_size = dest.size();
//     UpperClass::extractToVec(predicate, dest, path);
//     std::size_t new_size = dest.size();
//     const double stackHeight = stackHeights.back() - stackHeights.front();
//     for (unsigned r = 1; r < repeat_count; ++r) {
//         Vec<dim, double> v = Primitive<dim>::ZERO_VEC;
//         v.vert() += stackHeight * r;
//         for (std::size_t i = old_size; i < new_size; ++i) {
//             dest.push_back(Translation<dim>::compress(const_pointer_cast< GeometryObjectD<dim> >(dest[i]), v));
//         }
//     }
// }

template <typename UpperClass>
GeometryObject::Subtree MultiStackContainer<UpperClass>::getPathsTo(const GeometryObject& el,
                                                                    const PathHints* path) const {
    if (repeat_count == 0) return GeometryObject::Subtree();
    GeometryObject::Subtree result = UpperClass::getPathsTo(el, path);
    if (!result.empty()) {
        const std::size_t size = result.children.size();  // original size
        const double stackHeight = stackHeights.back() - stackHeights.front();
        for (unsigned r = 1; r < repeat_count; ++r)
            for (std::size_t org_child_no = 0; org_child_no < size; ++org_child_no) {
                auto& org_child = const_cast<Translation<UpperClass::DIM>&>(
                    static_cast<const Translation<UpperClass::DIM>&>(*(result.children[org_child_no].object)));
                shared_ptr<Translation<UpperClass::DIM>> new_child = org_child.copyShallow();
                new_child->translation[UpperClass::GROWING_DIR] += stackHeight;
                result.children.push_back(GeometryObject::Subtree(new_child, result.children[org_child_no].children));
            }
    }
    return result;
}

template <typename UpperClass>
GeometryObject::Subtree MultiStackContainer<UpperClass>::getPathsAt(
    const typename MultiStackContainer<UpperClass>::DVec& point,
    bool all) const {
    if (repeat_count == 0) return GeometryObject::Subtree();
    MultiStackContainer::DVec new_point = point;
    if (!reduceHeight(new_point[UpperClass::GROWING_DIR])) return GeometryObject::Subtree();
    return UpperClass::getPathsAt(new_point, all);
}

template <typename UpperClass>
bool MultiStackContainer<UpperClass>::contains(const typename MultiStackContainer<UpperClass>::DVec& p) const {
    if (repeat_count == 0) return false;
    DVec p_reduced = p;
    if (!reduceHeight(p_reduced[UpperClass::GROWING_DIR])) return false;
    return UpperClass::contains(p_reduced);
}

template <typename UpperClass>
shared_ptr<Material> MultiStackContainer<UpperClass>::getMaterial(
    const typename MultiStackContainer<UpperClass>::DVec& p) const {
    if (repeat_count == 0) return shared_ptr<Material>();
    DVec p_reduced = p;
    if (!reduceHeight(p_reduced[UpperClass::GROWING_DIR])) return shared_ptr<Material>();
    return UpperClass::getMaterial(p_reduced);
}

template <typename UpperClass>
shared_ptr<GeometryObject> MultiStackContainer<UpperClass>::getChildNo(std::size_t child_no) const {
    if (child_no >= getChildrenCount()) {
        auto children_count = getChildrenCount();
        if (children_count == 0) throw OutOfBoundsException("getChildNo", "child_no", child_no, "nothing", "nothing");
        throw OutOfBoundsException("getChildNo", "child_no", child_no, 0, children_count - 1);
    }
    if (child_no < children.size()) return children[child_no];
    auto result = children[child_no % children.size()]->copyShallow();
    result->translation[UpperClass::GROWING_DIR] +=
        double(child_no / children.size()) * (stackHeights.back() - stackHeights.front());
    return result;
}

template <typename UpperClass> std::size_t MultiStackContainer<UpperClass>::getRealChildrenCount() const {
    return UpperClass::getChildrenCount();
}

template <typename UpperClass>
shared_ptr<GeometryObject> MultiStackContainer<UpperClass>::getRealChildNo(std::size_t child_no) const {
    return UpperClass::getChildNo(child_no);
}

template <typename UpperClass>
void MultiStackContainer<UpperClass>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    UpperClass::writeXMLAttr(dest_xml_object, axes);
    dest_xml_object.attr(REPEAT_ATTR, repeat_count);
}

template <typename StackContainerT>
static inline void addChild(
    StackContainerT& result,
    const StackContainerT& src,
    std::size_t child_no,
    typename std::vector<std::pair<shared_ptr<typename StackContainerT::ChildType>, Vec<3, double>>>&
        children_after_change) {
    result.addUnsafe(children_after_change[child_no].first, src.getAlignerAt(child_no));
}

// it has to overload generic version above, but it is for shelf and so doesn't use aligners
static inline void addChild(
    MultiStackContainer<ShelfContainer2D>& result,
    const MultiStackContainer<ShelfContainer2D>& /*src*/,
    std::size_t child_no,
    typename std::vector<std::pair<shared_ptr<typename ShelfContainer2D::ChildType>, Vec<3, double>>>&
        children_after_change) {
    result.addUnsafe(children_after_change[child_no].first);
}

template <typename StackContainerT>
static inline void addChild(StackContainerT& result,
                            const StackContainerT& src,
                            const shared_ptr<GeometryObject>& child,
                            std::size_t child_no) {
    result.addUnsafe(static_pointer_cast<typename StackContainerT::ChildType>(child), src.getAlignerAt(child_no));
}

// it has to overload generic version above, but it is for shelf and so doesn't use aligners
static inline void addChild(MultiStackContainer<ShelfContainer2D>& result,
                            const MultiStackContainer<ShelfContainer2D>& /*src*/,
                            const shared_ptr<GeometryObject>& child,
                            std::size_t /*child_no*/) {
    result.addUnsafe(static_pointer_cast<MultiStackContainer<ShelfContainer2D>::ChildType>(child));
}

template <typename UpperClass> shared_ptr<GeometryObject> MultiStackContainer<UpperClass>::shallowCopy() const {
    shared_ptr<MultiStackContainer<UpperClass>> result =
        plask::make_shared<MultiStackContainer<UpperClass>>(this->repeat_count, this->getBaseHeight());
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        addChild(*result, *this, children[child_no]->getChild(), child_no);
    return result;
}

template <typename UpperClass>
shared_ptr<GeometryObject> MultiStackContainer<UpperClass>::deepCopy(
    std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const {
    auto found = copied.find(this);
    if (found != copied.end()) return found->second;
    shared_ptr<MultiStackContainer<UpperClass>> result =
        plask::make_shared<MultiStackContainer<UpperClass>>(this->repeat_count, this->getBaseHeight());
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (children[child_no]->getChild())
            addChild(*result, *this, children[child_no]->getChild()->deepCopy(copied), child_no);
    copied[this] = result;
    return result;
}

template <typename UpperClass>
shared_ptr<GeometryObject> MultiStackContainer<UpperClass>::changedVersionForChildren(
    std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change,
    Vec<3, double>* /*recomended_translation*/) const {
    shared_ptr<MultiStackContainer<UpperClass>> result =
        plask::make_shared<MultiStackContainer<UpperClass>>(this->repeat_count, this->getBaseHeight());
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (children_after_change[child_no].first) addChild(*result, *this, child_no, children_after_change);
    return result;
}

template <typename UpperClass>
void MultiStackContainer<UpperClass>::addPointsAlongToSet(std::set<double>& points,
                                                     Primitive<3>::Direction direction,
                                                     unsigned max_steps,
                                                     double min_step_size) const {
    if (repeat_count == 0) return;
    if (repeat_count == 1 || direction != UpperClass::GROWING_DIR + (3 - DIM)) {
        UpperClass::addPointsAlongToSet(points, direction, max_steps, min_step_size);
        return;
    }
    double shift = stackHeights.back() - stackHeights.front();
    std::set<double> points0;
    UpperClass::addPointsAlongToSet(points0, direction, max_steps, min_step_size);
    for (size_t i = 0; i < repeat_count; ++i) {
        double trans = i * shift;
        for (double p : points0) points.insert(p + trans);
    }
}

template <typename UpperClass>
void MultiStackContainer<UpperClass>::addLineSegmentsToSet(
    std::set<typename GeometryObjectD<UpperClass::DIM>::LineSegment>& segments,
    unsigned max_steps,
    double min_step_size) const {
    if (repeat_count == 0) return;
    if (repeat_count == 1) {
        UpperClass::addLineSegmentsToSet(segments, max_steps, min_step_size);
        return;
    }
    typedef typename GeometryObjectD<MultiStackContainer<UpperClass>::DIM>::LineSegment LineSegment;
    std::set<LineSegment> segments0;
    UpperClass::addLineSegmentsToSet(segments0, max_steps, min_step_size);
    DVec shift = Primitive<DIM>::ZERO_VEC;
    shift[size_t(UpperClass::GROWING_DIR)] = stackHeights.back() - stackHeights.front();
    for (size_t i = 0; i < repeat_count; ++i) {
        DVec trans = i * shift;
        for (const auto& s : segments0) segments.insert(LineSegment(s.p0() + trans, s.p1() + trans));
    }
}

template class PLASK_API MultiStackContainer<StackContainer<2>>;
template class PLASK_API MultiStackContainer<StackContainer<3>>;
template class PLASK_API MultiStackContainer<ShelfContainer2D>;

/// Helper used by read_... stack functions.
struct HeightReader {
    XMLReader& reader;
    const char* what;
    int whereWasZero;
    double alignZero;
    bool align;

    HeightReader(XMLReader& reader, const char* what)
        : reader(reader),
          what(what),
          whereWasZero(reader.hasAttribute(BASEH_ATTR) ? -2 : -1),
          alignZero(0.),
          align(false) {}

    inline void setZero(shared_ptr<plask::GeometryObject> stack) {
        if (whereWasZero != -1) throw XMLException(reader, format("{} shift has already been specified.", what));
        whereWasZero = int(stack->getRealChildrenCount());
    }

    bool tryReadZeroTag(shared_ptr<plask::GeometryObject> stack) {
        if (reader.getNodeName() != "zero") return false;
        setZero(stack);
        reader.requireTagEnd();
        return true;
    }

    void tryReadZeroAttr(shared_ptr<plask::GeometryObject> stack) {
        auto zeroAttr = reader.getAttribute<double>(ZERO_ATTR);
        if (!zeroAttr) return;
        setZero(stack);
        alignZero = *zeroAttr;
        align = true;
    }

    template <typename StackPtrT> void setBaseHeight(StackPtrT stack, bool reverse) {
        if (whereWasZero >= 0) {
            if (align)
                stack->alignZeroOn(reverse ? stack->getRealChildrenCount() - whereWasZero - 1 : whereWasZero,
                                   alignZero);
            else
                stack->setZeroBefore(reverse ? stack->getRealChildrenCount() - whereWasZero : whereWasZero);
        }
    }
};

template <int dim> static shared_ptr<GeometryObject> read_StackContainer(GeometryReader& reader) {
    HeightReader height_reader(reader.source, "Stack's vertical");
    const double baseH = reader.source.getAttribute(BASEH_ATTR, 0.0);

    shared_ptr<StackContainer<dim>> result(
        reader.source.hasAttribute(REPEAT_ATTR)
            ? new MultiStackContainer<StackContainer<dim>>(reader.source.getAttribute(REPEAT_ATTR, 1u), baseH)
            : new StackContainer<dim>(baseH));

    result->default_aligner =
        align::fromXML(reader.source, reader.getAxisNames(), StackContainer<dim>::DefaultAligner());

    GeometryReader::SetExpectedSuffix suffixSetter(
        reader, dim == 2 ? PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D : PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    read_children(
        reader,
        [&]() -> PathHints::Hint {
            height_reader.tryReadZeroAttr(result);
            auto aligner = align::fromXML(reader.source, reader.getAxisNames(), result->default_aligner);
            return result->push_front(reader.readExactlyOneChild<typename StackContainer<dim>::ChildType>(), aligner);
        },
        [&]() {
            if (height_reader.tryReadZeroTag(result)) return;
            result->push_front(reader.readObject<typename StackContainer<dim>::ChildType>());
        });
    height_reader.setBaseHeight(result, true);
    return result;
}

static GeometryReader::RegisterObjectReader stack2D_reader(PLASK_STACK2D_NAME, read_StackContainer<2>);
static GeometryReader::RegisterObjectReader stack3D_reader(PLASK_STACK3D_NAME, read_StackContainer<3>);

static shared_ptr<GeometryObject> read_ShelfContainer2D(GeometryReader& reader) {
    HeightReader height_reader(reader.source, "Shelf's horizontal");
    // TODO migrate to gap which can update self
    shared_ptr<Gap1D<2, Primitive<2>::DIRECTION_TRAN>> total_size_gap;  // gap which can change total size
    double required_total_size;  // required total size, valid only if total_size_gap is not nullptr
    const double baseH = reader.source.getAttribute(BASEH_ATTR, 0.0);
    shared_ptr<ShelfContainer2D> result(
        reader.source.hasAttribute(REPEAT_ATTR)
            ? new MultiStackContainer<ShelfContainer2D>(reader.source.getAttribute(REPEAT_ATTR, 1u), baseH)
            : new ShelfContainer2D(baseH));
    bool requireEqHeights = reader.source.getAttribute(REQUIRE_EQUAL_HEIGHTS_ATTR, true);
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    read_children(
        reader,
        [&]() {
            height_reader.tryReadZeroAttr(result);
            return result->push_back(reader.readExactlyOneChild<typename ShelfContainer2D::ChildType>());
        },
        [&]() {
            if (height_reader.tryReadZeroTag(result)) return;
            shared_ptr<Gap1D<2, Primitive<2>::DIRECTION_TRAN>> this_gap;
            if (reader.source.getNodeName() == Gap1D<2, Primitive<2>::DIRECTION_TRAN>::NAME) {
                plask::optional<double> total_size_attr = reader.source.getAttribute<double>("total");
                if (total_size_attr) {  // total size provided?
                    if (total_size_gap) throw XMLException(reader.source, "total size has been already chosen.");
                    required_total_size = *total_size_attr;
                    total_size_gap = this_gap = static_pointer_cast<Gap1D<2, Primitive<2>::DIRECTION_TRAN>>(
                        static_pointer_cast<Translation<2>>(result->addGap(0.0).second)->getChild());
                } else {
                    this_gap = static_pointer_cast<Gap1D<2, Primitive<2>::DIRECTION_TRAN>>(
                        static_pointer_cast<Translation<2>>(
                            result
                                ->addGap(reader.source.requireAttribute<double>(
                                    Gap1D<2, Primitive<2>::DIRECTION_TRAN>::XML_SIZE_ATTR))
                                .second)
                            ->getChild());
                }
                reader.registerObjectNameFromCurrentNode(this_gap);
                reader.source.requireTagEnd();
                return;
            }
            result->push_back(reader.readObject<ShelfContainer2D::ChildType>());
        });
    if (total_size_gap) {
        if (required_total_size < result->getHeight()) {
            reader.manager.throwErrorIfNotDraft(
                XMLException(reader.source, "required total width of shelf is lower than sum of children widths"));
            total_size_gap->setSize(0);
        } else
            total_size_gap->setSize(required_total_size - result->getHeight());
    }
    height_reader.setBaseHeight(result, false);
    if (requireEqHeights) {
        try {
            result->ensureFlat();
        } catch (const Exception& e) {
            reader.manager.throwErrorIfNotDraft(XMLException(reader.source, e.what()));
        }
    }
    return result;
}

static GeometryReader::RegisterObjectReader horizontalstack_reader(PLASK_SHELF_NAME, read_ShelfContainer2D);
static GeometryReader::RegisterObjectReader horizontalstack2D_reader(
    PLASK_SHELF_NAME PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D,
    read_ShelfContainer2D);

}  // namespace plask
