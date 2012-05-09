#include "stack.h"

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
    this->fireChanged(GeometryElement::Event::RESIZE|GeometryElement::Event::CHILD_LIST);
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
    children.erase(children.begin() + index);
    stackHeights.erase(stackHeights.begin() + index);
    updateAllHeights(index);
}

/*template <typename PredicateT>
bool remove_if(PredicateT predicate) {
    std::deque<shared_ptr<TranslationT>> deleted;
    auto dst = children.begin();
    for (auto i: children)
        if (predicate(i)) deleted.push_back(i);
        else *dst++ = i;
    children.erase(dst, children.end());
    updateAllHeights();
    for (auto i: deleted)
        disconnectOnChildChanged(*i);
    if (deleted.size() != 0) {
        this->fireChildrenChanged();
        return true;
    } else
        return false;
}

virtual bool removeT(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate);*/

template <int dim, int growingDirection>
bool StackContainerBaseImpl<dim, growingDirection>::removeIfTUnsafe(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
    if (GeometryElementContainer<dim>::removeIfTUnsafe(predicate)) {
        this->rebuildStackHeights();
        this->fireChildrenChanged();
        return true;
    } else
        return false;
}

template class StackContainerBaseImpl<2, Primitive<2>::DIRECTION_UP>;
template class StackContainerBaseImpl<3, Primitive<3>::DIRECTION_UP>;
template class StackContainerBaseImpl<2, Primitive<2>::DIRECTION_TRAN>;

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
    connectOnChildChanged(*trans_geom);
    children.insert(children.begin() + pos, trans_geom);
    aligners.insert(aligners.begin() + pos, aligner.cloneUnique());
    stackHeights.insert(stackHeights.begin() + pos, stackHeights[pos]);
    const double delta = bb.upper.up - bb.lower.up;
    for (std::size_t i = pos + 1; i < children.size(); ++i) {
        stackHeights[i] += delta;
        children[i]->translation.up += delta;
    }
    stackHeights.back() += delta;
    this->fireChildrenChanged();
    return PathHints::Hint(shared_from_this(), trans_geom);
}

template <int dim>
void StackContainer<dim>::setAlignerAt(std::size_t child_nr, const Aligner& aligner) {
    this->ensureIsValidChildNr(child_nr, "setAlignerAt");
    if (aligners[child_nr].get() == &aligner) return; //protected against self assigment
    aligners[child_nr] = aligner.cloneUnique();
    aligners[child_nr]->align(*children[child_nr]);
    this->fireChanged(GeometryElement::Event::RESIZE);
}

template <int dim>
bool StackContainer<dim>::removeIfTUnsafe(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate) {
    auto dst = children.begin();
    auto al_dst = aligners.begin();
    auto al_src = aligners.begin();
    for (auto i: children) {
        if (predicate(i))
            disconnectOnChildChanged(*i);
        else {
            *dst++ = i;
            *al_dst++ = std::move(*al_src);
        }
        ++al_src;
    }
    if (dst != children.end()) {
        children.erase(dst, children.end());
        aligners.erase(al_dst, aligners.end());
        return true;
    } else
        return false;
}

template <int dim>
void StackContainer<dim>::removeAtUnsafe(std::size_t index) {
    children.erase(children.begin() + index);
    stackHeights.erase(stackHeights.begin() + index);
    aligners.erase(aligners.begin() + index);
    this->updateAllHeights(index);
}


template class StackContainer<2>;
template class StackContainer<3>;


bool HorizontalStack::allChildrenHaveSameHeights() const {
    if (children.size() < 2) return true;
    double height = children.front()->getBoundingBoxSize().up;
    for (std::size_t i = 1; i < children.size(); ++i)
        if (height != children[i]->getBoundingBoxSize().up)
            return false;
    return true;
}

PathHints::Hint HorizontalStack::addUnsafe(const shared_ptr<ChildType>& el) {
    double el_translation, next_height;
    auto elBB = el->getBoundingBox();
    calcHeight(elBB, stackHeights.back(), el_translation, next_height);
    shared_ptr<TranslationT> trans_geom = make_shared<TranslationT>(el, vec(el_translation, -elBB.lower.up));
    connectOnChildChanged(*trans_geom);
    children.push_back(trans_geom);
    stackHeights.push_back(next_height);
    this->fireChildrenChanged();
    return PathHints::Hint(shared_from_this(), trans_geom);
}

#define baseH_attr "from"
#define repeat_attr "repeat"

shared_ptr<GeometryElement> read_StackContainer2d(GeometryReader& reader) {
    const double baseH = reader.source.getAttribute(baseH_attr, 0.0);
    std::unique_ptr<align::Aligner2d<align::DIRECTION_TRAN>> default_aligner(
          align::fromStr<align::DIRECTION_TRAN>(reader.source.getAttribute<std::string>(reader.getAxisTranName(), "l")));

    shared_ptr< StackContainer<2> > result(
                    reader.source.hasAttribute(repeat_attr) ?
                    new MultiStackContainer<2>(reader.source.getAttribute(repeat_attr, 1), baseH) :
                    new StackContainer<2>(baseH)
                );
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    read_children<StackContainer<2>>(reader,
            [&]() {
                boost::optional<std::string> aligner_str = reader.source.getAttribute(reader.getAxisTranName());
                if (aligner_str) {
                   std::unique_ptr<align::Aligner2d<align::DIRECTION_TRAN>> aligner(align::fromStr<align::DIRECTION_TRAN>(*aligner_str));
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

shared_ptr<GeometryElement> read_StackContainer3d(GeometryReader& reader) {
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

static GeometryReader::RegisterElementReader stack2d_reader("stack" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D, read_StackContainer2d);
static GeometryReader::RegisterElementReader stack3d_reader("stack" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D, read_StackContainer3d);

}   // namespace plask
