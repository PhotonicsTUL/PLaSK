#include "stack.h"
#include "separator.h"

#define baseH_attr "shift"
#define repeat_attr "repeat"
#define require_equal_heights_attr "flat"

namespace plask {

template <int dim, typename Primitive<dim>::Direction growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::setBaseHeight(double newBaseHeight) {
    if (getBaseHeight() == newBaseHeight) return;
    double diff = newBaseHeight - getBaseHeight();
    stackHeights.front() = newBaseHeight;
    for (std::size_t i = 1; i < stackHeights.size(); ++i) {
        stackHeights[i] += diff;
        children[i-1]->translation[growingDirection] += diff;
        //children[i-1]->fireChanged(GeometryObject::Event::RESIZE);
    }
    this->fireChildrenChanged();    //TODO should this be called? children was moved but not removed/inserted
    //this->fireChanged(GeometryObject::Event::RESIZE);
    //TODO: block connection to not react on children changed, call children[i]->fireChanged(GeometryObject::Event::RESIZE); for each child, call this->fireChanged(GeometryObject::Event::RESIZE delegate;
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::setZeroHeightBefore(std::size_t index)
{
    std::size_t h_count = stackHeights.size();
    if (index >= h_count)
        throw OutOfBoundsException("setZeroHeightBefore", "index", index, 0, h_count-1);
    setBaseHeight(stackHeights[0] - stackHeights[index]);
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
std::size_t StackContainerBaseImpl<dim, growingDirection>::getInsertionIndexForHeight(double height) const {
    return std::lower_bound(stackHeights.begin(), stackHeights.end(), height) - stackHeights.begin();
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
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

template <int dim, typename Primitive<dim>::Direction growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::removeAtUnsafe(std::size_t index) {
    GeometryObjectContainer<dim>::removeAtUnsafe(index);
    stackHeights.pop_back();
    updateAllHeights(index);
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
void StackContainerBaseImpl<dim, growingDirection>::writeXMLAttr(XMLWriter::Element &dest_xml_object, const AxisNames &) const {
    dest_xml_object.attr(baseH_attr, getBaseHeight());
}

template <int dim, typename Primitive<dim>::Direction growingDirection>
bool StackContainerBaseImpl<dim, growingDirection>::removeIfTUnsafe(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
    if (GeometryObjectContainer<dim>::removeIfTUnsafe(predicate)) {
        this->rebuildStackHeights();
        return true;
    } else
        return false;
}

template struct StackContainerBaseImpl<2, Primitive<2>::DIRECTION_VERT>;
template struct StackContainerBaseImpl<3, Primitive<3>::DIRECTION_VERT>;
template struct StackContainerBaseImpl<2, Primitive<2>::DIRECTION_TRAN>;

/*template <int dim>    //this is fine but GeometryObjects doesn't have copy constructors at all, because signal doesn't have copy constructor
StackContainer<dim>::StackContainer(const StackContainer& to_copy)
    : StackContainerBaseImpl<dim>(to_copy) //copy all but aligners
{
    std::vector<Aligner*> aligners_copy;
    aligners_copy.reserve(to_copy.size());
    for (auto a: to_copy.aligners) aligners_copy.push_back(a->clone());
    this->aligners = aligners_copy;
}*/

template <>
const StackContainer<2>::ChildAligner& StackContainer<2>::DefaultAligner() {
    static auto leftZeroAl = align::left(0);
    return leftZeroAl;
}

template <>
const StackContainer<3>::ChildAligner& StackContainer<3>::DefaultAligner() {
    static auto leftBackAl = align::left(0) & align::back(0);
    return leftBackAl;
}

template <int dim>
PathHints::Hint StackContainer<dim>::insertUnsafe(const shared_ptr<ChildType>& el, const std::size_t pos, const ChildAligner& aligner) {
    const auto bb = el->getBoundingBox();
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
    this->fireChildrenInserted(pos, pos+1);
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

template <int dim>
void StackContainer<dim>::removeAtUnsafe(std::size_t index) {
    GeometryObjectContainer<dim>::removeAtUnsafe(index);
    this->aligners.erase(this->aligners.begin() + index);
    stackHeights.pop_back();
    this->updateAllHeights(index);
}

template <int dim>
void StackContainer<dim>::writeXML(XMLWriter::Element &parent_xml_object, GeometryObject::WriteXMLCallback &write_cb, AxisNames axes) const {
    XMLWriter::Element container_tag = write_cb.makeTag(parent_xml_object, *this, axes);
    if (GeometryObject::WriteXMLCallback::isRef(container_tag)) return;
    this->writeXMLAttr(container_tag, axes);
    for (int i = children.size()-1; i >= 0; --i) {   //children are written in reverse order
        XMLWriter::Element child_tag = write_cb.makeChildTag(container_tag, *this, i);
        writeXMLChildAttr(child_tag, i, axes);
        children[i]->getChild()->writeXML(child_tag, write_cb, axes);
    }
}

template <int dim>
void StackContainer<dim>::writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const {
    this->aligners[child_index].writeToXML(dest_xml_child_tag, axes);
}

template <int dim>
shared_ptr<GeometryObject> StackContainer<dim>::changedVersionForChildren(std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const {
    shared_ptr< StackContainer<dim> > result = make_shared< StackContainer<dim> >(this->getBaseHeight());
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (children_after_change[child_no].first)
            result->addUnsafe(children_after_change[child_no].first, this->aligners[child_no]);
    return result;
}

template struct StackContainer<2>;
template struct StackContainer<3>;

PathHints::Hint ShelfContainer2D::addGap(double size) {
    return addUnsafe(make_shared<Gap1D<2, Primitive<2>::DIRECTION_TRAN>>(size));
}

bool ShelfContainer2D::isFlat() const {
    std::size_t first = 0;
    while (first < children.size() && children[first]->getChildType() == GeometryObject::TYPE_SEPARATOR)
        ++first;
    if (first >= children.size()-2)
        return true;    //(almost) same separators
    const double height = children[first]->getBoundingBoxSize().vert();
    for (std::size_t i = first+1; i < children.size(); ++i)
        if (children[i]->getChildType() != GeometryObject::TYPE_SEPARATOR &&
            height != children[i]->getBoundingBoxSize().vert())
            return false;
    return true;
}

PathHints::Hint ShelfContainer2D::addUnsafe(const shared_ptr<ChildType>& el) {
    double el_translation, next_height;
    auto elBB = el->getBoundingBox();
    calcHeight(elBB, stackHeights.back(), el_translation, next_height);
    shared_ptr<TranslationT> trans_geom = make_shared<TranslationT>(el, vec(el_translation, -elBB.lower.vert()));
    connectOnChildChanged(*trans_geom);
    children.push_back(trans_geom);
    stackHeights.push_back(next_height);
    this->fireChildrenInserted(children.size()-1, children.size());
    return PathHints::Hint(shared_from_this(), trans_geom);
}

PathHints::Hint ShelfContainer2D::insertUnsafe(const shared_ptr<ChildType>& el, const std::size_t pos) {
    const auto bb = el->getBoundingBox();
    shared_ptr<TranslationT> trans_geom = make_shared<TranslationT>(el, vec(stackHeights[pos] - bb.lower.tran(), -bb.lower.vert()));
    connectOnChildChanged(*trans_geom);
    children.insert(children.begin() + pos, trans_geom);
    stackHeights.insert(stackHeights.begin() + pos, stackHeights[pos]);
    const double delta = bb.upper.tran() - bb.lower.tran();
    for (std::size_t i = pos + 1; i < children.size(); ++i) {
        stackHeights[i] += delta;
        children[i]->translation.tran() += delta;
    }
    stackHeights.back() += delta;
    this->fireChildrenInserted(pos, pos+1);
    return PathHints::Hint(shared_from_this(), trans_geom);
}

shared_ptr<GeometryObject> ShelfContainer2D::changedVersionForChildren(std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const {
    shared_ptr< ShelfContainer2D > result = make_shared< ShelfContainer2D >(this->getBaseHeight());
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (children_after_change[child_no].first)
            result->addUnsafe(children_after_change[child_no].first);
    return result;
}

void ShelfContainer2D::writeXMLAttr(XMLWriter::Element &dest_xml_object, const AxisNames &) const {
    dest_xml_object.attr(require_equal_heights_attr, false);
}


template <int dim>
bool MultiStackContainer<dim>::reduceHeight(double& height) const {
    const double zeroBasedStackHeight = stackHeights.back() - stackHeights.front();
    const double zeroBasedRequestHeight = height - stackHeights.front();
    if (zeroBasedRequestHeight < 0.0 || zeroBasedRequestHeight > zeroBasedStackHeight * repeat_count)
        return false;
    height = std::fmod(zeroBasedRequestHeight, zeroBasedStackHeight) + stackHeights.front();
    return true;
}

//TODO good but unused
/*template <int dim>
bool MultiStackContainer<dim>::intersects(const Box& area) const {
    const double minusZeroBasedStackHeight = stackHeights.front() - stackHeights.back();
    for (unsigned r = 0; r < repeat_count; ++r)
        if (UpperClass::intersects(area.translatedUp(minusZeroBasedStackHeight*r)))
            return true;
    return false;
}*/

template <int dim>
void MultiStackContainer<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path) const {
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
void MultiStackContainer<dim>::getObjectsToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObject> >& dest, const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->shared_from_this());
        return;
    }
    std::size_t old_size = dest.size();
    UpperClass::getObjectsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (unsigned r = 1; r < repeat_count; ++r)
        for (std::size_t i = old_size; i < new_size; ++i)
            dest.push_back(dest[i]);
}

template <int dim>
void MultiStackContainer<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path) const {
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
            dest.back().vert() += stackHeight * r;
        }
}

// template <int dim>
// void MultiStackContainer<dim>::extractToVec(const GeometryObject::Predicate &predicate, std::vector< shared_ptr<const GeometryObjectD<dim> > >& dest, const PathHints *path) const {
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

template <int dim>
GeometryObject::Subtree MultiStackContainer<dim>::getPathsTo(const GeometryObject& el, const PathHints* path) const {
    GeometryObject::Subtree result = UpperClass::getPathsTo(el, path);
    if (!result.empty()) {
        const std::size_t size = result.children.size();   // original size
        const double stackHeight = stackHeights.back() - stackHeights.front();
        for (unsigned r = 1; r < repeat_count; ++r)
            for (std::size_t org_child_no = 0; org_child_no < size; ++org_child_no) {
                auto& org_child = const_cast<Translation<dim>&>(static_cast<const Translation<dim>&>(*(result.children[org_child_no].object)));
                shared_ptr<Translation<dim>> new_child = org_child.copyShallow();
                new_child->translation.vert() += stackHeight;
                result.children.push_back(GeometryObject::Subtree(new_child, result.children[org_child_no].children));
            }
    }
    return result;
}

template <int dim>
GeometryObject::Subtree MultiStackContainer<dim>::getPathsAt(const MultiStackContainer::DVec &point, bool all) const {
    MultiStackContainer::DVec new_point = point;
    reduceHeight(new_point.vert());
    return GeometryObjectContainer<dim>::getPathsAt(new_point, all);
}

template class MultiStackContainer<2>;
template class MultiStackContainer<3>;

template <int dim>
shared_ptr<GeometryObject> MultiStackContainer<dim>::getChildNo(std::size_t child_no) const {
    if (child_no >= getChildrenCount()) throw OutOfBoundsException("getChildNo", "child_no", child_no, 0, getChildrenCount()-1);
    if (child_no < children.size()) return children[child_no];
    auto result = children[child_no % children.size()]->copyShallow();
    result->translation.vert() += (child_no / children.size()) * (stackHeights.back() - stackHeights.front());
    return result;
}

template <int dim>
void MultiStackContainer<dim>::writeXMLAttr(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const {
    StackContainer<dim>::writeXMLAttr(dest_xml_object, axes);
    dest_xml_object.attr(repeat_attr, repeat_count);
}

template <int dim>
shared_ptr<GeometryObject> MultiStackContainer<dim>::changedVersionForChildren(std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const {
    shared_ptr< MultiStackContainer<dim> > result = make_shared< MultiStackContainer<dim> >(this->repeat_count, this->getBaseHeight());
    for (std::size_t child_no = 0; child_no < children.size(); ++child_no)
        if (children_after_change[child_no].first)
            result->addUnsafe(children_after_change[child_no].first, this->getAlignerAt(child_no));
    return result;
}

/// Helper used by read_... stack functions.
struct HeightReader {
    XMLReader& reader;
    int whereWasZeroTag;

    HeightReader(XMLReader& reader): reader(reader), whereWasZeroTag(reader.hasAttribute(baseH_attr) ? -2 : -1) {}

    bool tryReadZero(shared_ptr<plask::GeometryObject> stack) {
        if (reader.getNodeName() != "zero") return false;
        if (whereWasZeroTag != -1) throw XMLException(reader, "Base height has been already chosen.");
        reader.requireTagEnd();
        whereWasZeroTag = stack->getRealChildrenCount();
        return true;
    }

    template <typename StackPtrT>
    void setBaseHeight(StackPtrT stack, bool reverse) {
        if (whereWasZeroTag >= 0) stack->setZeroHeightBefore(reverse ? stack->getRealChildrenCount() - whereWasZeroTag : whereWasZeroTag);
    }
};

template <int dim>
static shared_ptr<GeometryObject> read_StackContainer(GeometryReader& reader) {
    HeightReader height_reader(reader.source);
    const double baseH = reader.source.getAttribute(baseH_attr, 0.0);

    shared_ptr<StackContainer<dim>> result(
                    reader.source.hasAttribute(repeat_attr) ?
                    new MultiStackContainer<dim>(reader.source.getAttribute(repeat_attr, 1), baseH) :
                    new StackContainer<dim>(baseH)
                );

    result->default_aligner = align::fromXML(reader.source, reader.getAxisNames(), StackContainer<dim>::DefaultAligner());

    GeometryReader::SetExpectedSuffix suffixSetter(reader, dim == 2 ? PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D : PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    read_children(reader,
            [&]() -> PathHints::Hint {
                auto aligner = align::fromXML(reader.source, reader.getAxisNames(), result->default_aligner);
                return result->push_front(reader.readExactlyOneChild<typename StackContainer<dim>::ChildType>(), aligner);
            },
            [&]() {
                if (height_reader.tryReadZero(result)) return;
                result->push_front(reader.readObject<typename StackContainer<dim>::ChildType>());
            }
    );
    height_reader.setBaseHeight(result, true);
    return result;
}

static GeometryReader::RegisterObjectReader stack2D_reader(StackContainer<2>::NAME, read_StackContainer<2>);
static GeometryReader::RegisterObjectReader stack3D_reader(StackContainer<3>::NAME, read_StackContainer<3>);

static shared_ptr<GeometryObject> read_ShelfContainer2D(GeometryReader& reader) {
    HeightReader height_reader(reader.source);
    //TODO migrate to gap which can update self
    shared_ptr<Gap1D<2, Primitive<2>::DIRECTION_TRAN>> total_size_gap;  //gap which can change total size
    double required_total_size;  //required total size, valid only if total_size_gap is not nullptr
    shared_ptr< ShelfContainer2D > result(new ShelfContainer2D(reader.source.getAttribute(baseH_attr, 0.0)));
    bool requireEqHeights = reader.source.getAttribute(require_equal_heights_attr, true);
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    read_children(reader,
            [&]() {
                return result->push_back(reader.readExactlyOneChild< typename ShelfContainer2D::ChildType >());
            },
            [&]() {
                if (height_reader.tryReadZero(result)) return;
                shared_ptr<Gap1D<2, Primitive<2>::DIRECTION_TRAN>> this_gap;
                if (reader.source.getNodeName() == Gap1D<2, Primitive<2>::DIRECTION_TRAN>::NAME) {
                    boost::optional<double> total_size_attr = reader.source.getAttribute<double>("total");
                    if (total_size_attr) {  //total size provided?
                        if (total_size_gap)
                            throw XMLException(reader.source, "Total size has been already chosen.");
                        required_total_size = *total_size_attr;
                        total_size_gap = this_gap =
                                static_pointer_cast<Gap1D<2, Primitive<2>::DIRECTION_TRAN>>(
                                     static_pointer_cast<Translation<2>>(result->addGap(0.0).second)->getChild()
                                );
                    } else {
                        this_gap =
                        static_pointer_cast<Gap1D<2, Primitive<2>::DIRECTION_TRAN>>(
                            static_pointer_cast<Translation<2>>(
                                result->addGap(reader.source.requireAttribute<double>(Gap1D<2, Primitive<2>::DIRECTION_TRAN>::XML_SIZE_ATTR)).second
                            )->getChild()
                        );
                    }
                    reader.registerObjectNameFromCurrentNode(this_gap);
                    reader.source.requireTagEnd();
                    return;
                }
                result->push_back(reader.readObject< ShelfContainer2D::ChildType >());
            }
    );
    if (total_size_gap) {
        if (required_total_size < result->getHeight())
            throw Exception("Required total width of shelf is lower than sum of children and gaps widths.");
        total_size_gap->setSize(required_total_size - result->getHeight());
    }
    height_reader.setBaseHeight(result, false);
    if (requireEqHeights) result->ensureFlat();
    return result;
}

static GeometryReader::RegisterObjectReader horizontalstack_reader(ShelfContainer2D::NAME, read_ShelfContainer2D);
static GeometryReader::RegisterObjectReader horizontalstack2D_reader("shelf" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D, read_ShelfContainer2D);

}   // namespace plask
