#ifndef PLASK__GEOMETRY_CONTAINER_H
#define PLASK__GEOMETRY_CONTAINER_H

/** @file
This file includes containers of geometries objects.
*/

#include <vector>
#include <algorithm>
#include <cmath>    //fmod
#include "../utils/metaprog.h"

#include "path.h"
#include "align.h"
#include "reader.h"
#include "../manager.h"
#include "../utils/string.h"

namespace plask {

/**
 * Template of base class for all container nodes.
 * Container nodes can include one or more child nodes with translations.
 *
 * @tparam dim GeometryObjectContainer dimension
 * @ingroup GEOMETRY_OBJ
 */
template <int dim>
struct GeometryObjectContainer: public GeometryObjectD<dim> {

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectContainer<dim>::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectContainer<dim>::Box Box;

    /// Type of the container children.
    typedef GeometryObjectD<dim> ChildType;

    /// Type of translation geometry object in space of this.
    typedef Translation<dim> TranslationT;

    /// Type of the vector holding container children
    typedef std::vector< shared_ptr<TranslationT> > TranslationVector;

protected:
    TranslationVector children;

    /**
     * Add attributes to child tag.
     *
     * Default implementation do nothing.
     * @param dest_xml_child_tag destination, child tag
     * @param child_index index of child
     * @param axes names of axes
     */
    virtual void writeXMLChildAttr(XMLWriter::Element &dest_xml_child_tag, std::size_t child_index, const AxisNames &axes) const;

    /**
     * This is called by changedVersion method to create new version of this container which consists with new children.
     * @param[in] children_after_change vector of new children
     * @param[out] recomended_translation optional, place to store recommended translation (if is not nullptr, it has all coordinates equals to 0.0)
     * @return copy of this, with new children
     */
    virtual shared_ptr<GeometryObject> changedVersionForChildren(std::vector<std::pair<shared_ptr<ChildType>, Vec<3, double>>>& children_after_change, Vec<3, double>* recomended_translation) const = 0;

public:

    /**
     * Call writeXMLAttr for this container attribute and writeXMLChildAttr for each child tag.
     * @param parent_xml_object
     * @param write_cb
     * @param axes
     */
    virtual void writeXML(XMLWriter::Element& parent_xml_object, GeometryObject::WriteXMLCallback& write_cb, AxisNames axes) const;

    // TODO container should reduce number of generated event from child if have 2 or more same children, for each children should be connected once

    /// Disconnect onChildChanged from current child change signal
    ~GeometryObjectContainer() {
        for (auto& c: children) disconnectOnChildChanged(*c);
    }

    /// Called by child.change signal, call this change
    virtual void onChildChanged(const GeometryObject::Event& evt) {
        this->fireChanged(evt.oryginalSource(), evt.flagsForParent());
    }

    /// Connect onChildChanged to current child change signal
    void connectOnChildChanged(Translation<dim>& child) {
        child.changedConnectMethod(this, &GeometryObjectContainer::onChildChanged);
    }

    /// Disconnect onChildChanged from current child change signal
    void disconnectOnChildChanged(Translation<dim>& child) {
        child.changedDisconnectMethod(this, &GeometryObjectContainer::onChildChanged);
    }

    /**
     * Get phisicaly stored children (with translations).
     * @return vector of translations object, each include children
     */
    const TranslationVector& getChildrenVector() const {
        return children;
    }

    /// @return GE_TYPE_CONTAINER
    virtual GeometryObject::Type getType() const { return GeometryObject::TYPE_CONTAINER; }

    virtual bool includes(const DVec& p) const {
        for (auto child: children) if (child->includes(p)) return true;
        return false;
    }

    virtual bool intersects(const Box& area) const {
        for (auto child: children) if (child->intersects(area)) return true;
        return false;
    }

    virtual Box getBoundingBox() const;

    /**
     * Iterate over children in reverse order and check if any returns material.
     * @return material of first child which returns non @c nullptr or @c nullptr if all children return @c nullptr
     */
    virtual shared_ptr<Material> getMaterial(const DVec& p) const;

    /*virtual void getLeafsInfoToVec(std::vector< std::tuple<shared_ptr<const GeometryObject>, Box, DVec> >& dest, const PathHints* path = 0) const {
        if (path) {
            auto c = path->getTranslationChildren<dim>(*this);
            if (!c.empty()) {
                for (auto child: c) child->getLeafsInfoToVec(dest, path);
                return;
            }
        }
        for (auto child: children) child->getLeafsInfoToVec(dest, path);
    }*/

    virtual void getBoundingBoxesToVec(const GeometryObject::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const;

    virtual void getObjectsToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const GeometryObject> >& dest, const PathHints* path = 0) const;

    /*virtual void getLeafsToVec(std::vector< shared_ptr<const GeometryObject> >& dest) const {
        for (auto child: children) child->getLeafsToVec(dest);
    }*/

    virtual void getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path = 0) const;

    // void extractToVec(const GeometryObject::Predicate &predicate, std::vector< shared_ptr<const GeometryObjectD<dim> > >& dest, const PathHints *path) const;

    /*virtual std::vector< std::tuple<shared_ptr<const GeometryObject>, DVec> > getLeafsWithTranslations() const {
        std::vector< std::tuple<shared_ptr<const GeometryObject>, DVec> > result;
        for (auto child: children) {
            std::vector< std::tuple<shared_ptr<const GeometryObject>, DVec> > child_leafs_tran = child->getLeafsWithTranslations();
            result.insert(result.end(), child_leafs_tran.begin(), child_leafs_tran.end());
        }
        return result;
    }*/

    virtual bool isInSubtree(const GeometryObject& el) const;

    template <typename ChildIter>
    GeometryObject::Subtree findPathsFromChildTo(ChildIter childBegin, ChildIter childEnd, const GeometryObject& el, const PathHints* path = 0) const {
        GeometryObject::Subtree result;
        for (auto child_iter = childBegin; child_iter != childEnd; ++child_iter) {
            GeometryObject::Subtree child_path = (*child_iter)->getPathsTo(el, path);
            if (!child_path.empty())
                result.children.push_back(std::move(child_path));
        }
        if (!result.children.empty())
            result.object = this->shared_from_this();
        return result;
    }

    virtual GeometryObject::Subtree getPathsTo(const GeometryObject& el, const PathHints* path = 0) const;

    virtual GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const;

    virtual std::size_t getChildrenCount() const { return children.size(); }

    virtual shared_ptr<GeometryObject> getChildAt(std::size_t child_nr) const {
        this->ensureIsValidChildNr(child_nr);
        return children[child_nr];
    }

    virtual shared_ptr<TranslationT> getTranslationOfRealChildAt(std::size_t child_nr) const {
        this->ensureIsValidChildNr(child_nr);
        return children[child_nr];
    }

    virtual shared_ptr<const GeometryObject> changedVersion(const GeometryObject::Changer& changer, Vec<3, double>* translation = 0) const;

    /**
     * Remove all children which fulfil predicate.
     *
     * This is unsafe but fast version, it doesn't call fireChildrenChanged() to inform listeners about this object changes.
     * Caller should do this manually or call removeIfT instead.
     * @param predicate returns true only if the child (with translation) passed as an argument should be deleted
     * @return true if anything has been removed
     */
    virtual bool removeIfTUnsafe(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate);

    /**
     * Remove all children which fulfil predicate.
     * @param predicate returns true only if the child (with translation) passed as an argument should be deleted
     * @return true if anything has been removed
     */
    bool removeIfT(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate);

    /**
     * Remove all children which fulfil predicate.
     * @tparam PredicateT functor which can take child as argument and return something convertable to bool
     * @param predicate returns true only if the child passed as an argument should be deleted
     * @return true if anything has been removed
     */
    bool removeIf(const std::function<bool(const shared_ptr<ChildType>& c)>& predicate) {
        return removeIfT([&](const shared_ptr<const TranslationT>& c) { return predicate(c->getChild()); });
    }

    /**
     * Remove all children exactly equal to @a el.
     * @param el child(ren) to remove
     * @return true if anything has been removed
     */
    bool removeT(shared_ptr<const TranslationT> el) {
        return removeIfT([&el](const shared_ptr<const TranslationT>& c) { return c == el; });
    }

    /**
     * Remove all children exactly equal to @a el.
     * @param el child(ren) to remove
     * @return true if anything has been removed
     */
    bool remove(shared_ptr<const ChildType> el) {
        return removeIf([&el](const shared_ptr<const ChildType>& c) { return c == el; });
    }

    /**
     * Remove child pointed, for this container, in @a hints.
     * @param hints path hints, see @ref geometry_paths
     * @return true if anything has been removed
     */
    bool remove(const PathHints& hints) {
        auto cset = hints.getChildren(*this);
        return removeIfT([&](const shared_ptr<TranslationT>& t) {
                       return cset.find(static_pointer_cast<GeometryObject>(t)) != cset.end();
                });
    }

    /**
     * Remove child at given @p index.
     *
     * This is unsafe but fast version, it doesn't check index and doesn't call fireChildrenChanged() to inform listeners about this object changes.
     * Caller should do this manually or call removeAt(std::size_t) instead.
     * @param index index of real child to remove
     */
    virtual void removeAtUnsafe(std::size_t index) {
        disconnectOnChildChanged(*children[index]);
        children.erase(children.begin() + index);
    }

};

/**
 * Read children, call child_param_read if children is in \<child\> tag and without_child_param_read if there was no \<child\> tag.
 * Read "path" parameter from each \<child\> tag.
 * @param reader reader
 * @param child_param_read functor called for each \<child\> tag, without parameters, should create child, add it to container and return PathHints::Hint (and move reader.source to end of current, \<child\>, tag)
 * @param without_child_param_read functor called for each children (when there was no \<child\> tag), should create child, add it to container, and move reader.source to end of current tag
 */
template <typename ChildParamF, typename WithoutChildParamF>
inline void read_children(GeometryReader& reader, ChildParamF child_param_read, WithoutChildParamF without_child_param_read) {
    while (reader.source.requireTagOrEnd()) {
        if (reader.source.getNodeName() == "child") {
            boost::optional<std::string> paths_str = reader.source.getAttribute("path");
            PathHints::Hint hint = child_param_read();  //this call readExactlyOneChild
            if (paths_str) {
                auto paths = splitEscIterator(*paths_str, ',');
                for (auto& path: paths) {
                    BadId::throwIfBad("path", path, '-');
                    reader.manager.pathHints[path].addHint(hint);
                }
            }
        } else {
            without_child_param_read();
        }
    }
}



}	// namespace plask

#endif // PLASK__GEOMETRY_CONTAINER_H
