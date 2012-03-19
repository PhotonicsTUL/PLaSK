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

    typedef std::vector< shared_ptr< Translation<dim> > > TranslationVector;

    ///Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::DVec DVec;

    ///Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::Rect Rect;

protected:
    TranslationVector children;

    /**
     * Remove all children which fulfil predicate.
     * @param predicate return true only if child passed as argument should be deleted
     * @tparam PredicateT functor which can take child as argument and return something convertable to bool
     */
    template <typename PredicateT>
    void removeAll(PredicateT predicate) {
        children.erease(
            std::remove_if(children.begin(), children.end(), predicate), children.end()
        );
    }

public:

    /**
     * Get phisicaly stored children (with translations).
     * @return vector of translations object, each include children
     */
    const TranslationVector& getChildrenVector() const {
        return children;
    }

    ///@return GE_TYPE_CONTAINER
    virtual GeometryElement::Type getType() const { return GeometryElement::TYPE_CONTAINER; }

    virtual bool inside(const DVec& p) const {
        for (auto child: children) if (child->inside(p)) return true;
        return false;
    }

    virtual bool intersect(const Rect& area) const {
        for (auto child: children) if (child->intersect(area)) return true;
        return false;
    }

    virtual Rect getBoundingBox() const {
        if (children.empty()) return Rect(Primitive<dim>::ZERO_VEC, Primitive<dim>::ZERO_VEC);
        Rect result = children[0]->getBoundingBox();
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

    /*virtual void getLeafsInfoToVec(std::vector< std::tuple<shared_ptr<const GeometryElement>, Rect, DVec> >& dest, const PathHints* path = 0) const {
        if (path) {
            auto c = path->getTranslationChildren<dim>(*this);
            if (!c.empty()) {
                for (auto child: c) child->getLeafsInfoToVec(dest, path);
                return;
            }
        }
        for (auto child: children) child->getLeafsInfoToVec(dest, path);
    }*/

    virtual void getLeafsBoundingBoxesToVec(std::vector<Rect>& dest, const PathHints* path = 0) const {
        if (path) {
            auto c = path->getTranslationChildren<dim>(*this);
            if (!c.empty()) {
                for (auto child: c) child->getLeafsBoundingBoxesToVec(dest, path);
                return;
            }
        }
        for (auto child: children) child->getLeafsBoundingBoxesToVec(dest, path);
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


    virtual std::size_t getChildCount() const { return children.size(); }

    virtual shared_ptr<GeometryElement> getChildAt(std::size_t child_nr) const {
        if (child_nr >= children.size()) throw OutOfBoundException("getChildAt", "child_nr", child_nr, 0, children.size()-1);
        return children[child_nr];
    }

    virtual shared_ptr<const GeometryElement> changedVersion(const GeometryElement::Changer& changer, Vec<3, double>* translation = 0) const {
        shared_ptr<const GeometryElement> result(this->shared_from_this());
        if (changer.apply(result, translation) || children.empty()) return result;
        //if (translation) *translation = vec(0.0, 0.0, 0.0); //we can't recommend nothing special
        //TODO code... what with paths? add paths to changedVersion method
    }

};

/**
 * Geometry elements container in which every child has an associated translation vector.
 */
//TODO some implementation are naive, and can be done faster with some caches
template < int dim >
struct TranslationContainer: public GeometryElementContainer<dim> {

    ///Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::DVec DVec;

    ///Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::Rect Rect;

    ///Type of this child.
    typedef GeometryElementD<dim> ChildType;

    ///Type of translation geometry elment in space of this.
    typedef Translation<dim> TranslationT;

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
        children.push_back(trans_geom);
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

    /**
     * Remove all children which fulfil predicate.
     * @param predicate return true only if child passed as argument should be deleted
     * @tparam PredicateT functor which can take child as argument and return something convertable to bool
     */
    template <typename PredicateT>
    void remove(PredicateT predicate) {
        removeAll(predicate);
    }

    /**
     * Remove all children exactly equal to @a el.
     * @param el child(ren) to remove
     */
    void remove(const ChildType* el) {
        removeAll([&el](ChildType* c) { return c->child == el; });
    }

    /**
     * Remove child pointed, for this container, in @a hints.
     * @param hints path hints, see @ref geometry_paths
     */
    void remove(const PathHints& hints) {
        auto cset = hints.getChildren(this);
        removeAll([&](TranslationT t) { return cset.find(t) != cset.end; });
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
