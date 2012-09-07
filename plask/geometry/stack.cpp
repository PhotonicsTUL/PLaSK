#include "stack.h"

#define baseH_attr "from"
#define repeat_attr "repeat"
#define require_equal_heights_attr "flat"

namespace plask {

template <int dim, int growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::setBaseHeight(double newBaseHeight) {
    if (getBaseHeight() == newBaseHeight) return;
    double diff = newBaseHeight - getBaseHeight();
    stackHeights.front() = newBaseHeight;
    for (std::size_t i = 1; i < stackHeights.size(); ++i) {
        stackHeights[i] += diff;
        children[i-1]->translation.components[growingDirection] += diff;
        //children[i-1]->fireChanged(GeometryElement::Event::RESIZE);
    }
    this->fireChildrenChanged();    //TODO should this be called? children was moved but not removed/inserted
    //this->fireChanged(GeometryElement::Event::RESIZE);
    //TODO: block connection to not react on children changed, call children[i]->fireChanged(GeometryElement::Event::RESIZE); for each child, call this->fireChanged(GeometryElement::Event::RESIZE delegate;
}

template <int dim, int growingDirection>
std::size_t StackContainerBaseImpl<dim, growingDirection>::getInsertionIndexForHeight(double height) const {
    return std::lower_bound(stackHeights.begin(), stackHeights.end(), height) - stackHeights.begin();
}

template <int dim, int growingDirection>
const shared_ptr<typename StackContainerBaseImpl<dim, growingDirection>::TranslationT>
StackContainerBaseImpl<dim, growingDirection>::getChildForHeight(double height) const {
    auto it = std::lower_bound(stackHeights.begin(), stackHeights.end(), height);
    if (it == stackHeights.end()) return shared_ptr<TranslationT>();
    if (it == stackHeights.begin()) {
        if (height == stackHeights.front()) return children[0];
        else return shared_ptr<TranslationT>();
    }
    return children[it-stackHeights.begin()-1];
}

template <int dim, int growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::removeAtUnsafe(std::size_t index) {
    GeometryElementContainer<dim>::removeAtUnsafe(index);
    stackHeights.pop_back();
    updateAllHeights(index);
}

template <int dim, int growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::writeXMLAttr(XMLWriter::Element &dest_xml_element, const AxisNames &) const {
    dest_xml_element.attr(baseH_attr, getBaseHeight());
}

template <int dim, int growingDirection>
bool StackContainerBaseImpl<dim, growingDirection>::removeIfTUnsafe(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
    if (GeometryElementContainer<dim>::removeIfTUnsafe(predicate)) {
        this->rebuildStackHeights();
        return true;
    } else
        return false;
}

template struct StackContainerBaseImpl<2, Primitive<2>::DIRECTION_UP>;
template struct StackContainerBaseImpl<3, Primitive<3>::DIRECTION_UP>;
template struct StackContainerBaseImpl<2, Primitive<2>::DIRECTION_TRAN>;

/*template <int dim>    //this is fine but GeometryElements doesn't have copy constructors at all, becose signal doesn't have copy constructor
StackContainer<dim>::StackContainer(const StackContainer& to_copy)
    : StackContainerBaseImpl<dim>(to_copy) //copy all but aligners
{
    std::vector<Aligner*> aligners_copy;
    aligners_copy.reserve(to_copy.size());
    for (auto a: to_copy.aligners) aligners_copy.push_back(a->clone());
    this->aligners = aligners_copy;
}*/

template <int dim>
PathHints::Hint StackContainer<dim>::insertUnsafe(const shared_ptr<ChildType>& el, const std::size_t pos, const Aligner& aligner) {
    const auto bb = el->getBoundingBox();
    shared_ptr<TranslationT> trans_geom = newTranslation(el, aligner, stackHeights[pos] - bb.lower.up, bb);
    this->connectOnChildChanged(*trans_geom);
    children.insert(children.begin() + pos, trans_geom);
    aligners.insert(aligners.begin() + pos, aligner.cloneUnique());
    stackHeights.insert(stackHeights.begin() + pos, stackHeights[pos]);
    const double delta = bb.upper.up - bb.lower.up;
    for (std::size_t i = pos + 1; i < children.size(); ++i) {
        stackHeights[i] += delta;
        children[i]->translation.up += delta;
    }
    stackHeights.back() += delta;
    this->fireChildrenInserted(pos, pos+1);
    return PathHints::Hint(shared_from_this(), trans_geom);
}

template <int dim>
void StackContainer<dim>::setAlignerAt(std::size_t child_nr, const Aligner& aligner) {
    this->ensureIsValidChildNr(child_nr, "setAlignerAt");
    if (aligners[child_nr].get() == &aligner) return; //protected against self assigment
    aligners[child_nr] = aligner.cloneUnique();
    aligners[child_nr]->align(*children[child_nr]);
    children[child_nr]->fireChanged();
}

template <int dim>
bool StackContainer<dim>::removeIfTUnsafe(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
    auto dst = children.begin();
    auto al_dst = aligners.begin();
    auto al_src = aligners.begin();
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
        aligners.erase(al_dst, aligners.end());
        this->rebuildStackHeights();
        return true;
    } else
        return false;
}

template <int dim>
void StackContainer<dim>::removeAtUnsafe(std::size_t index) {
    GeometryElementContainer<dim>::removeAtUnsafe(index);
    aligners.erase(aligners.begin() + index);
    stackHeights.pop_back();
    this->updateAllHeights(index);
}

template <int dim>
void StackContainer<dim>::writeXML(XMLWriter::Element &parent_xml_element, GeometryElement::WriteXMLCallback &write_cb, AxisNames axes) const {
    XMLWriter::Element container_tag = write_cb.makeTag(parent_xml_element, *this, axes);
    if (GeometryElement::WriteXMLCallback::isRef(container_tag)) return;
    this->writeXMLAttr(container_tag, axes);
    for (int i = children.size()-1; i >= 0; --i) {   //children are written in reverse order
        XMLWriter::Element child_tag = write_cb.makeChildTag(container_tag, *this, i);
        writeXMLChildAttr(child_tag, i, axes);
        children[i]->getChild()->writeXML(child_tag, write_cb, axes);
    }
}

template <>
void StackContainer<2>::writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const {
    dest_xml_child_tag.attr(axes.getNameForTran(), aligners[child_index]->str());
}

template <>
void StackContainer<3>::writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const {
    //TODO
    //dest_xml_child_tag.attr(axes.getNameForLon(), aligners[child_index]->str());
    //dest_xml_child_tag.attr(axes.getNameForTran(), aligners[child_index]->str());
}

template struct StackContainer<2>;
template struct StackContainer<3>;


bool ShelfContainer2D::isFlat() const {
    if (children.size() < 2) return true;
    double height = children.front()->getBoundingBoxSize().up;
    for (std::size_t i = 1; i < children.size(); ++i)
        if (height != children[i]->getBoundingBoxSize().up)
            return false;
    return true;
}

PathHints::Hint ShelfContainer2D::addUnsafe(const shared_ptr<ChildType>& el) {
    double el_translation, next_height;
    auto elBB = el->getBoundingBox();
    calcHeight(elBB, stackHeights.back(), el_translation, next_height);
    shared_ptr<TranslationT> trans_geom = make_shared<TranslationT>(el, vec(el_translation, -elBB.lower.up));
    connectOnChildChanged(*trans_geom);
    children.push_back(trans_geom);
    stackHeights.push_back(next_height);
    this->fireChildrenInserted(children.size()-1, children.size());
    return PathHints::Hint(shared_from_this(), trans_geom);
}

PathHints::Hint ShelfContainer2D::insertUnsafe(const shared_ptr<ChildType>& el, const std::size_t pos) {
    const auto bb = el->getBoundingBox();
    shared_ptr<TranslationT> trans_geom = make_shared<TranslationT>(el, vec(stackHeights[pos] - bb.lower.tran, -bb.lower.up));
    connectOnChildChanged(*trans_geom);
    children.insert(children.begin() + pos, trans_geom);
    stackHeights.insert(stackHeights.begin() + pos, stackHeights[pos]);
    const double delta = bb.upper.tran - bb.lower.tran;
    for (std::size_t i = pos + 1; i < children.size(); ++i) {
        stackHeights[i] += delta;
        children[i]->translation.tran += delta;
    }
    stackHeights.back() += delta;
    this->fireChildrenInserted(pos, pos+1);
    return PathHints::Hint(shared_from_this(), trans_geom);
}


template <int dim>
const bool MultiStackContainer<dim>::reduceHeight(double& height) const {
    const double zeroBasedStackHeight = stackHeights.back() - stackHeights.front();
    const double zeroBasedRequestHeight = height - stackHeights.front();
    if (zeroBasedRequestHeight < 0.0 || zeroBasedRequestHeight > zeroBasedStackHeight * repeat_count)
        return false;
    height = std::fmod(zeroBasedRequestHeight, zeroBasedStackHeight) + stackHeights.front();
    return true;
}

template <int dim>
bool MultiStackContainer<dim>::intersect(const Box& area) const {
    const double minusZeroBasedStackHeight = stackHeights.front() - stackHeights.back();
    for (unsigned r = 0; r < repeat_count; ++r)
        if (UpperClass::intersect(area.translatedUp(minusZeroBasedStackHeight*r)))
            return true;
    return false;
}

template <int dim>
void MultiStackContainer<dim>::getBoundingBoxesToVec(const GeometryElement::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    std::size_t old_size = dest.size();
    UpperClass::getBoundingBoxesToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    const double stackHeight = stackHeights.back() - stackHeights.front();
    for (unsigned r = 1; r < repeat_count; ++r) {
        for (std::size_t i = old_size; i < new_size; ++i)
            dest.push_back(dest[i]);
        for (auto i = dest.end() - (new_size-old_size); i != dest.end(); ++i)
            i->translateUp(stackHeight * r);
    }
}

template <int dim>
void MultiStackContainer<dim>::getElementsToVec(const GeometryElement::Predicate& predicate, std::vector< shared_ptr<const GeometryElement> >& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->shared_from_this());
        return;
    }
    std::size_t old_size = dest.size();
    UpperClass::getElementsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (unsigned r = 1; r < repeat_count; ++r)
        for (std::size_t i = old_size; i < new_size; ++i)
            dest.push_back(dest[i]);
}

template <int dim>
void MultiStackContainer<dim>::getPositionsToVec(const GeometryElement::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    std::size_t old_size = dest.size();
    UpperClass::getPositionsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    const double stackHeight = stackHeights.back() - stackHeights.front();
    for (unsigned r = 1; r < repeat_count; ++r)
        for (std::size_t i = old_size; i < new_size; ++i) {
            dest.push_back(dest[i]);
            dest.back().up += stackHeight * r;
        }
}

template <int dim>
GeometryElement::Subtree MultiStackContainer<dim>::getPathsTo(const GeometryElement& el, const PathHints* path) const {
    GeometryElement::Subtree result = UpperClass::getPathsTo(el, path);
    if (!result.empty()) {
        const std::size_t size = result.children.size();   //oryginal size
        const double stackHeight = stackHeights.back() - stackHeights.front();
        for (unsigned r = 1; r < repeat_count; ++r)
            for (std::size_t org_child_nr = 0; org_child_nr < size; ++org_child_nr) {
                auto& org_child = const_cast<Translation<dim>&>(static_cast<const Translation<dim>&>(*(result.children[org_child_nr].element)));
                shared_ptr<Translation<dim>> new_child = org_child.copyShallow();
                new_child->translation.up += stackHeight;
                result.children.push_back(GeometryElement::Subtree(new_child, result.children[org_child_nr].children));
            }
    }
    return result;
}

template <int dim>
GeometryElement::Subtree MultiStackContainer<dim>::getPathsTo(const MultiStackContainer::DVec &point) const {
    MultiStackContainer::DVec new_point = point;
    new_point.up = reduceHeight(new_point.up);
    return GeometryElementContainer<dim>::getPathsTo(new_point);
}

template class MultiStackContainer<2>;
template class MultiStackContainer<3>;

template <int dim>
shared_ptr<GeometryElement> MultiStackContainer<dim>::getChildAt(std::size_t child_nr) const {
    if (child_nr >= getChildrenCount()) throw OutOfBoundException("getChildAt", "child_nr", child_nr, 0, getChildrenCount()-1);
    if (child_nr < children.size()) return children[child_nr];
    auto result = children[child_nr % children.size()]->copyShallow();
    result->translation.up += (child_nr / children.size()) * (stackHeights.back() - stackHeights.front());
    return result;
}

template <int dim>
void MultiStackContainer<dim>::writeXMLAttr(XMLWriter::Element &dest_xml_element, const AxisNames &axes) const {
    StackContainer<dim>::writeXMLAttr(dest_xml_element, axes);
    dest_xml_element.attr(repeat_attr, repeat_count);
}

shared_ptr<GeometryElement> read_StackContainer2D(GeometryReader& reader) {
    const double baseH = reader.source.getAttribute(baseH_attr, 0.0);
    std::unique_ptr<align::Aligner2D<align::DIRECTION_TRAN>> default_aligner(
          align::fromStr<align::DIRECTION_TRAN>(reader.source.getAttribute<std::string>(reader.getAxisTranName(), "l")));

    shared_ptr< StackContainer<2> > result(
                    reader.source.hasAttribute(repeat_attr) ?
                    new MultiStackContainer<2>(reader.source.getAttribute(repeat_attr, 1), baseH) :
                    new StackContainer<2>(baseH)
                );
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    read_children<StackContainer<2>>(reader,
            [&]() -> PathHints::Hint {
                boost::optional<std::string> aligner_str = reader.source.getAttribute(reader.getAxisTranName());
                if (aligner_str) {
                   std::unique_ptr<align::Aligner2D<align::DIRECTION_TRAN>> aligner(align::fromStr<align::DIRECTION_TRAN>(*aligner_str));
                   return result->push_front(reader.readExactlyOneChild< typename StackContainer<2>::ChildType >(), *aligner);
                } else {
                   return result->push_front(reader.readExactlyOneChild< typename StackContainer<2>::ChildType >(), *default_aligner);
                }
            },
            [&](const shared_ptr<typename StackContainer<2>::ChildType>& child) {
                result->push_front(child);
            }
    );
    return result;
}

shared_ptr<GeometryElement> read_StackContainer3D(GeometryReader& reader) {
    const double baseH = reader.source.getAttribute(baseH_attr, 0.0);
    //TODO default aligner (see above)
    shared_ptr< StackContainer<3> > result(
                    reader.source.hasAttribute(repeat_attr) ?
                    new MultiStackContainer<3>(reader.source.getAttribute(repeat_attr, 1), baseH) :
                    new StackContainer<3>(baseH)
                );
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    read_children<StackContainer<3>>(reader,
            [&]() {
                return result->push_front(reader.readExactlyOneChild< typename StackContainer<3>::ChildType >(),
                                          align::fromStr<align::DIRECTION_LON, align::DIRECTION_TRAN>(
                                              reader.source.getAttribute<std::string>(reader.getAxisLonName(), "b"),
                                              reader.source.getAttribute<std::string>(reader.getAxisTranName(), "l")
                                          ));
            },
            [&](const shared_ptr<typename StackContainer<3>::ChildType>& child) {
                result->push_front(child);
            }
    );
    return result;
}

static GeometryReader::RegisterElementReader stack2D_reader(StackContainer<2>::NAME, read_StackContainer2D);
static GeometryReader::RegisterElementReader stack3D_reader(StackContainer<3>::NAME, read_StackContainer3D);

shared_ptr<GeometryElement> read_ShelfContainer2D(GeometryReader& reader) {
    shared_ptr< ShelfContainer2D > result(new ShelfContainer2D(reader.source.getAttribute(baseH_attr, 0.0)));
    bool requireEqHeights = reader.source.getAttribute(require_equal_heights_attr, false);
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    read_children<StackContainer<2>>(reader,
            [&]() {
                return result->push_back(reader.readExactlyOneChild< typename ShelfContainer2D::ChildType >());
            },
            [&](const shared_ptr<typename ShelfContainer2D::ChildType>& child) {
                result->push_back(child);
            }
    );
    if (requireEqHeights) result->ensureFlat();
    return result;
}

static GeometryReader::RegisterElementReader horizontalstack_reader(ShelfContainer2D::NAME, read_ShelfContainer2D);
static GeometryReader::RegisterElementReader horizontalstack2D_reader("shelf" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D, read_ShelfContainer2D);

}   // namespace plask
