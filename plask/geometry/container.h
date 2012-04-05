#ifndef PLASK__GEOMETRY_CONTAINER_H
#define PLASK__GEOMETRY_CONTAINER_H

/** @file
This file includes containers of geometries elements.
*/

#include <vector>
#include <algorithm>
#include <cmath>    //fmod
#include "../utils/metaprog.h"

#include "path.h"
#include "align.h"
#include "reader.h"

namespace plask {

/**
 * Template of base class for all container nodes.
 * Container nodes can include one or more child nodes with translations.
 *
 * @tparam dim GeometryElementContainer dimension
 */
template <int dim>
struct GeometryElementContainer: public GeometryElementD<dim> {

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::Box Box;

    /// Type of the container children.
    typedef GeometryElementD<dim> ChildType;

    /// Type of translation geometry element in space of this.
    typedef Translation<dim> TranslationT;

    /// Type of the vector holiding container children
    typedef std::vector<shared_ptr<TranslationT>> TranslationVector;

protected:
    TranslationVector children;

    void ensureIsValidChildNr(std::size_t child_nr, const char* method_name = "getChildAt", const char* arg_name = "child_nr") const {
        if (child_nr >= children.size())
            throw OutOfBoundException(method_name, arg_name, child_nr, 0, children.size()-1);
    }

    /// Inform observers that children list was changed (also that this is resized)
    void fireChildrenChanged() {
        this->fireChanged(GeometryElement::Event::RESIZE | GeometryElement::Event::CHILD_LIST);
    }

public:

    // TODO container should reduce number of generated event from child if have 2 or more same children, for each children should be connected once

    /// Called by child.change signal, call this change
    virtual void onChildChanged(const GeometryElement::Event& evt) {
        this->fireChanged(evt.flagsForParent());
    }

    /// Connect onChildChanged to current child change signal
    void connectOnChildChanged(Translation<dim>& child) {
        child.changed.connect(
            boost::bind(&GeometryElementContainer::onChildChanged, this, _1)
        );
    }

    /// Disconnect onChildChanged from current child change signal
    void disconnectOnChildChanged(Translation<dim>& child) {
        child.changed.disconnect(
            boost::bind(&GeometryElementContainer::onChildChanged, this, _1)
        );
    }

    /**
     * Get phisicaly stored children (with translations).
     * @return vector of translations object, each include children
     */
    const TranslationVector& getChildrenVector() const {
        return children;
    }

    /// @return GE_TYPE_CONTAINER
    virtual GeometryElement::Type getType() const { return GeometryElement::TYPE_CONTAINER; }

    virtual bool inside(const DVec& p) const {
        for (auto child: children) if (child->inside(p)) return true;
        return false;
    }

    virtual bool intersect(const Box& area) const {
        for (auto child: children) if (child->intersect(area)) return true;
        return false;
    }

    virtual Box getBoundingBox() const {
        if (children.empty()) return Box(Primitive<dim>::ZERO_VEC, Primitive<dim>::ZERO_VEC);
        Box result = children[0]->getBoundingBox();
        for (std::size_t i = 1; i < children.size(); ++i)
            result.include(children[i]->getBoundingBox());
        return result;
    }

    /**
     * Iterate over children in reverse order and check if any returns material.
     * @return material of first child which returns non @c nullptr or @c nullptr if all children return @c nullptr
     */
    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        for (auto child_it = children.rbegin(); child_it != children.rend(); ++child_it) {
            shared_ptr<Material> r = (*child_it)->getMaterial(p);
            if (r != nullptr) return r;
        }
        return shared_ptr<Material>();
    }

    /*virtual void getLeafsInfoToVec(std::vector< std::tuple<shared_ptr<const GeometryElement>, Box, DVec> >& dest, const PathHints* path = 0) const {
        if (path) {
            auto c = path->getTranslationChildren<dim>(*this);
            if (!c.empty()) {
                for (auto child: c) child->getLeafsInfoToVec(dest, path);
                return;
            }
        }
        for (auto child: children) child->getLeafsInfoToVec(dest, path);
    }*/

    virtual void getBoundingBoxesToVec(const GeometryElement::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const {
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

    virtual void getLeafsToVec(std::vector< shared_ptr<const GeometryElement> >& dest) const {
        for (auto child: children) child->getLeafsToVec(dest);
    }

    virtual std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > getLeafsWithTranslations() const {
        std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > result;
        for (auto child: children) {
            std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > child_leafs_tran = child->getLeafsWithTranslations();
            result.insert(result.end(), child_leafs_tran.begin(), child_leafs_tran.end());
        }
        return result;
    }

    virtual bool isInSubtree(const GeometryElement& el) const {
        if (&el == this) return true;
        for (auto child: children)
            if (child->isInSubtree(el))
                return true;
        return false;
    }

    template <typename ChildIter>
    GeometryElement::Subtree findPathsFromChildTo(ChildIter childBegin, ChildIter childEnd, const GeometryElement& el, const PathHints* path = 0) const {
        GeometryElement::Subtree result;
        for (auto child_iter = childBegin; child_iter != childEnd; ++child_iter) {
            GeometryElement::Subtree child_path = (*child_iter)->findPathsTo(el, path);
            if (!child_path.empty())
                result.children.push_back(std::move(child_path));
        }
        if (!result.children.empty())
            result.element = this->shared_from_this();
        return result;
    }

    virtual GeometryElement::Subtree findPathsTo(const GeometryElement& el, const PathHints* path = 0) const {
        if (this == &el) return this->shared_from_this();
        if (path) {
            auto hintChildren = path->getTranslationChildren<dim>(*this);
            if (!hintChildren.empty())
                return findPathsFromChildTo(hintChildren.begin(), hintChildren.end(), el, path);
        }
        return findPathsFromChildTo(children.begin(), children.end(), el, path);
    }


    virtual std::size_t getChildrenCount() const { return children.size(); }

    virtual shared_ptr<GeometryElement> getChildAt(std::size_t child_nr) const {
        ensureIsValidChildNr(child_nr);
        return children[child_nr];
    }

    virtual shared_ptr<const GeometryElement> changedVersion(const GeometryElement::Changer& changer, Vec<3, double>* translation = 0) const {
        shared_ptr<const GeometryElement> result(this->shared_from_this());
        if (changer.apply(result, translation) || children.empty()) return result;
        //if (translation) *translation = vec(0.0, 0.0, 0.0); // we can't recommend nothing special
        //TODO code... what with paths? add paths to changedVersion method
    }

    /**
     * Remove all children which fulfil predicate.
     * @tparam PredicateT functor which can take child as argument and return something convertable to bool
     * @param predicate returns true only if the child passed as an argument should be deleted
     * @return true if anything has been removed
     */
    template <typename PredicateT>
    bool remove_if(PredicateT predicate) {
        bool removed = false;
        auto dst = children.begin();
        for (auto i: children)
            if (predicate(i)) { disconnectOnChildChanged(*i); removed = true; }
            else *dst++ = i;
        children.erase(dst, children.end());
        return removed;
    }

    /**
     * Remove all children exactly equal to @a el.
     * @param el child(ren) to remove
     * @return true if anything has been removed
     */
    bool remove(shared_ptr<const ChildType> el) {
        return remove_if([&el](const shared_ptr<const ChildType>& c) { return c == el; });
    }

    /**
     * Remove child pointed, for this container, in @a hints.
     * @param hints path hints, see @ref geometry_paths
     * @return true if anything has been removed
     */
    bool remove(const PathHints& hints) {
        auto cset = hints.getChildren(*this);
        return remove_if([&](shared_ptr<TranslationT> t) { return cset.find(static_pointer_cast<GeometryElement>(t)) != cset.end(); });
    }
};

/**
 * Geometry elements container in which every child has an associated translation vector.
 */
//TODO some implementation are naive, and can be done faster with some caches
template < int dim >
struct TranslationContainer: public GeometryElementContainer<dim> {

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::Box Box;

    /// Type of this child.
    typedef typename GeometryElementContainer<dim>::ChildType ChildType;

    /// Type of translation geometry element in space of this.
    typedef typename GeometryElementContainer<dim>::TranslationT TranslationT;

    using GeometryElementContainer<dim>::children;
    using GeometryElementContainer<dim>::shared_from_this;

    /**
     * Add new child (translated) to end of children vector.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el new child
     * @param translation trasnalation of child
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint addUnsafe(const shared_ptr<ChildType>& el, const DVec& translation = Primitive<dim>::ZERO_VEC) {
        shared_ptr<TranslationT> trans_geom(new TranslationT(el, translation));
        connectOnChildChanged(*trans_geom);
        children.push_back(trans_geom);
        this->fireChildrenChanged();
        return PathHints::Hint(shared_from_this(), trans_geom);
    }

    /**
     * Add new child (trasnlated) to end of children vector.
     * @param el new child
     * @param translation trasnalation of child
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(const shared_ptr<ChildType>& el, const DVec& translation = Primitive<dim>::ZERO_VEC) {
        this->ensureCanHasAsChild(*el);
        return addUnsafe(el, translation);
    }

};

/**
 * Read children, construct ConstructedType::ChildType for each, call child_param_read if children is in \<child\> tag.
 * Read "path" parameter from each \<child\> tag.
 * @param reader reader
 * @param child_param_read functor called for each \<child\> tag, without parameters, should create child, add it to container and return PathHints::Hint
 * @param without_child_param_add functor called for each children (when there was no \<child\> tag), as paremter it take one geometry element (child) of type ConstructedType::ChildType
 */
template <typename ConstructedType, typename ChildParamF, typename WithoutChildParamF>
inline void read_children(GeometryReader& reader, ChildParamF child_param_read, WithoutChildParamF without_child_param_add) {

    std::string container_tag_name = reader.source.getNodeName();

    while (reader.source.read()) {
        switch (reader.source.getNodeType()) {

            case irr::io::EXN_ELEMENT_END:
                if (reader.source.getNodeName() != container_tag_name)
                    throw XMLUnexpectedElementException("end of \"" + container_tag_name + "\" tag");
                return; // container has been read

            case irr::io::EXN_ELEMENT:
                if (reader.source.getNodeName() == std::string("child")) {
                    boost::optional<std::string> path = XML::getAttribute(reader.source, "path");
                    PathHints::Hint hint = child_param_read();
                    if (path)
                        reader.manager.pathHints[*path].addHint(hint);  //this call readExactlyOneChild
                } else {
                    without_child_param_add(reader.readElement< typename ConstructedType::ChildType >());
                }

            case irr::io::EXN_COMMENT:
                break;  //skip comments

            default:
                throw XMLUnexpectedElementException("<child> or geometry element tag");
        }
    }
    throw XMLUnexpectedEndException();
}



}	// namespace plask

#endif // PLASK__GEOMETRY_CONTAINER_H
