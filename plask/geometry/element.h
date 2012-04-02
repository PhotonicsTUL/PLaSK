#ifndef PLASK__GEOMETRY_ELEMENT_H
#define PLASK__GEOMETRY_ELEMENT_H

/** @file
This file includes base classes for geometries elements.
*/


#include <vector>
#include <tuple>

#include "../material/material.h"
#include "primitives.h"
#include "../utils/iterators.h"

#include <boost/signals2.hpp>

namespace plask {

struct PathHints;

/**
 * Transform coordinates of points between two geometries.
 *
 * Transform objects can be composed.
 */
struct GeometryTransform {
    //Vec3 to(Vec3)
    //Vec3 from(Vec3)
    //GeometryTransform compose(GeometryTransform)
};

/**
 * Base class for all geometries.
 */
//TODO generalized methods which are for leafs now, to use polimorphic predicate and allow to work not only for leafs
//getLeafsBoundingBoxes(...) -> getBoundingBoxes(..., IsLeaf/IsEqual/...)
struct GeometryElement: public enable_shared_from_this<GeometryElement> {

    ///Type of geometry element.
    enum Type {
        TYPE_LEAF = 0,         ///< leaf element (has no child)
        TYPE_TRANSFORM = 1,    ///< transform element (has one child)
        TYPE_SPACE_CHANGER = 2,///< transform element which changing its space, typically changing number of dimensions (has one child)
        TYPE_CONTAINER = 3     ///< container (can have more than one child)
    };

    /**
     * Store information about event connected with geometry element.
     *
     * Subclasses of this can includes additional information about specific type of event.
     */
    struct Event {

        enum Flags {
            DELETE = 1,     ///< is deleted
            RESIZE = 1<<1,     ///< size could changed
            CHILD = 1<<2    ///< delegated from child
        };

    private:
        GeometryElement& _source;
        unsigned char _flags;

    public:

        /**
         * Get event source.
         * @return event source
         */
        const GeometryElement& source() const { return _source; }

        /// Event flags
        unsigned char flags() const { return _flags; }
        unsigned char flagsWithout(unsigned char flagsToRemove) const { return _flags & ~flagsToRemove; }
        unsigned char flagsForParent() const { return flagsWithout(GeometryElement::Event::DELETE) | CHILD; }

        bool hasFlag(Flags flag) const { return _flags & flag; }
        bool isDelete() const { return hasFlag(DELETE); }
        bool isResize() const { return hasFlag(RESIZE); }
        bool isDelgatedFromChild() const { return hasFlag(CHILD); }

        Event(GeometryElement& source, unsigned char falgs): _source(source), _flags(falgs) {}
        virtual ~Event() {} //for eventual subclassing
    };

    /**
     * This structure can refer to part of geometry tree.
     */
    struct Subtree {

        ///Geometry element.
        shared_ptr<const GeometryElement> element;

        ///Some (but not necessary all) children of element.
        std::vector<Subtree> children;

        /**
         * Construct subtree witch is empty or has only one node.
         * @param element geometry element, or null pointer to construct empty Subtree
         */
        Subtree(shared_ptr<const GeometryElement> element = shared_ptr<const GeometryElement>()): element(element) {}

        /**
         * Construct subtree.
         * @param element geometry element
         * @param children some (but not necessary all) children of @p element
         */
        Subtree(shared_ptr<const GeometryElement> element, const std::vector<Subtree>& children): element(element), children(children) {}

        /**
         * Check if this subtree inludes more than one branch (has more than one children or has one child which has more than one branch).
         * @return @c true only if this subtree inludes branches, @c false if it is linear path
         */
        bool hasBranches() const;

        /**
         * Convert this subtree to linear path: element, child[0].element, child[0].child[0].element, ...
         *
         * Throw excpetion if this subtree is not linear path (inludes more than one branch).
         */
        std::vector<shared_ptr<const GeometryElement>> toLinearPath() const;

        /**
         * Check if this subtree is empty (its element points to null).
         * @return @c true only if this subtree is empty.
         */
        bool empty() const { !element; }
    };

    struct Changer {

        /**
         * Try to apply changes.
         * @param to_change[in,out] pointer to element which eventualy will be changed (in such case pointer after call can point to another geometry element)
         * @param translation[out] optional, extra translation for element after change (in case of 2d object caller reads only \a tran and \a up components of this vector)
         * @return @c true only if something was changed, @c false if nothing was changed (in such case changer doesn't change arguments)
         */
        virtual bool apply(shared_ptr<const GeometryElement>& to_change, Vec<3, double>* translation = 0) const = 0;

    };

    struct CompositeChanger: public Changer {

        std::vector<const Changer*> changers;

        CompositeChanger(const Changer* changer);

        CompositeChanger& operator()(const Changer* changer);

        ///Delete all holded changers
        ~CompositeChanger();

        virtual bool apply(shared_ptr<const GeometryElement>& to_change, Vec<3, double>* translation = 0) const;

    };

    struct ReplaceChanger: public Changer {

        shared_ptr<const GeometryElement> from, to;
        Vec<3, double> translation;

        ReplaceChanger() {}

        ReplaceChanger(const shared_ptr<const GeometryElement>& from, const shared_ptr<const GeometryElement>& to, Vec<3, double> translation)
            : from(from), to(to), translation(translation) {}

        template <typename F>
        ReplaceChanger(const shared_ptr<const GeometryElement>&, F calc_replace): from(from) {
            this->to = calc_replace(this->from, &this->translation);
        }

        virtual bool apply(shared_ptr<const GeometryElement>& to_change, Vec<3, double>* translation = 0) const;

    };

    struct ToBlockChanger: ReplaceChanger {

        ToBlockChanger(const shared_ptr<const GeometryElement>& toChange, const shared_ptr<Material>& material);

    };

    /// Changed signal, fired when element was changed.
    boost::signals2::signal<void(const Event&)> changed;

    template<typename EventT = Event, typename ...Args>
    void fireChanged(Args&&... params) {
        changed(EventT(*this, std::forward<Args>(params)...));
    }

    /**
     * Virtual destructor. Inform all change listeners.
     */
    virtual ~GeometryElement();

    /**
     * Check if geometry is: leaf, transform or container type element.
     * @return type of this element
     */
    virtual Type getType() const = 0;

    bool isLeaf() const { return getType() == TYPE_LEAF; }
    bool isTransform() const { return getType() == TYPE_TRANSFORM || getType() == TYPE_SPACE_CHANGER; }
    bool isSpaceChanger() const { return getType() == TYPE_SPACE_CHANGER; }
    bool isContainer() const { return getType() == TYPE_CONTAINER; }

    /**
     * Get number of dimentions.
     * @return number of dimentions
     */
    virtual int getDimensionsCount() const = 0;

    /**
     * Check if element is ready for calculation.
     * Throw exception if element is in bad state and can't be used in calculations, for example has not required children, etc.
     * Default implementation do nothing, but inherited class can change this behavior.
     * @throw Exception if element is not ready for calculation
     */
    virtual void validate() const {}

    /**
     * Check if @a el is in subtree with @c this in root.
     * @param el element to search for
     * @return @c true only if @a el is in subtree with @c this in root
     */
    virtual bool isInSubtree(const GeometryElement& el) const = 0;

    /**
     * Find paths to @a el.
     * @param el element to search for
     * @param pathHints (optional) path hints which limits search space
     * @return sub-tree with paths to given element (@p el is in all leafs), empty sub-tree if @p el is not in subtree with @c this in root
     */
    virtual Subtree findPathsTo(const GeometryElement& el, const PathHints* pathHints = 0) const = 0;

    /**
     * Append all leafs in subtree with this in root to vector @p dest.
     * @param dest leafs destination vector
     */
    virtual void getLeafsToVec(std::vector< shared_ptr<const GeometryElement> >& dest) const = 0;

    /**
     * Get all leafs in subtree with this object as root.
     * @return all leafs in subtree with this object as root
     */
    std::vector< shared_ptr<const GeometryElement> > getLeafs() const {
        std::vector< shared_ptr<const GeometryElement> > result;
        getLeafsToVec(result);
        return result;
    }

    /**
     * Get number of element children in geometry graph.
     * @return number of children
     */
    virtual std::size_t getChildrenCount() const = 0;

    /**
     * Get child with given index.
     * @param child_nr index of child to get
     * @return child with index @p child_nr
     */
    virtual shared_ptr<GeometryElement> getChildAt(std::size_t child_nr) const = 0;

    /**
     * Get number of real (physicaly stored) children in geometry graph.
     *
     * By default call getChildrenCount(), but elements of some types (like multi-stack) redefine this.
     * @return number of real children
     */
    virtual std::size_t getRealChildCount() const;

    /**
     * Get real (physicaly stored) child with given index.
     *
     * By default call getChildAt(child_nr), but elements of some types (like multi-stack) redefine this.
     * @param child_nr index of real child to get
     * @return child with index @p child_nr
     */
    virtual shared_ptr<GeometryElement> getRealChildAt(std::size_t child_nr) const;

private:
    struct ChildGetter {    //used by begin(), end()
        shared_ptr<const GeometryElement> el;
        ChildGetter(const shared_ptr<const GeometryElement>& el): el(el) {}
        shared_ptr<GeometryElement> operator()(std::size_t index) const { return el->getChildAt(index); }
    };

public:

    ///@return begin begin iterator over children
    FunctorIndexedIterator<ChildGetter> begin() const {
        return FunctorIndexedIterator<ChildGetter>(ChildGetter(this->shared_from_this()), 0);
    }

    ///@return end end iterator over children
    FunctorIndexedIterator<ChildGetter> end() const {
        return FunctorIndexedIterator<ChildGetter>(ChildGetter(this->shared_from_this()), getChildrenCount());
    }

    //virtual GeometryTransform getTransform()

    /**
     * Get this or copy of this with some changes in subtree.
     * @param changer[in] changer which will be aplied to subtree with this in root
     * @param translation[out] recommended translation of this after change
     * @return pointer to this (if nothing was change) or copy of this with some changes in subtree
     */
    virtual shared_ptr<const GeometryElement> changedVersion(const Changer& changer, Vec<3, double>* translation = 0) const = 0;

protected:

    /**
     * Throw CyclicReferenceException if @p potential_parent is in subtree with this in root.
     * @param potential_parent[in] potential, new parent of this
     */
    void ensureCanHasAsParent(const GeometryElement& potential_parent) const;

    /**
     * Throw CyclicReferenceException if @p potential_child has this in subtree.
     * @param potential_child[in] potential, new child of this
     */
    void ensureCanHasAsChild(const GeometryElement& potential_child) const { potential_child.ensureCanHasAsParent(*this); }

};

/**
 * Template of base classes for geometry elements in space with given number of dimensions (2 or 3).
 * @tparam dimensions number of dimensions, 2 or 3
 */
template < int dimensions >
struct GeometryElementD: public GeometryElement {

    static const int dim = dimensions;
    typedef typename Primitive<dim>::Box Box;
    typedef typename Primitive<dim>::DVec DVec;

    int getDimensionsCount() const { return dimensions; }

    //virtual Box getBoundingBox() const;

    /**
     * Check if geometry includes point.
     * @param p point
     * @return true only if this geometry includes point @a p
     */
    virtual bool inside(const DVec& p) const = 0;

    /**
     * Check if geometry includes some point from given @a area.
     * @param area rectangular area
     * @return true only if this geometry includes some points from @a area
     */
    virtual bool intersect(const Box& area) const = 0;

    /**
     * Calculate minimal rectangle which includes all points of geometry element.
     * @return calculated rectangle
     */
    virtual Box getBoundingBox() const = 0;

    virtual DVec getBoundingBoxSize() const { return getBoundingBox().size(); }

    /**
     * Return material in a given point inside the geometry element
     * @param p point
     * @return material in given point, or @c nullptr if this GeometryElement not includes point @a p
     */
    virtual shared_ptr<Material> getMaterial(const DVec& p) const = 0;

    //virtual std::vector<Material*> getMaterials(Mesh);        ??

    //virtual void getLeafsInfoToVec(std::vector<std::tuple<shared_ptr<const GeometryElement>, Box, DVec>>& dest, const PathHints* path = 0) const = 0;

    /**
     * Calculate and append to vector bounding boxes of all leafs, optionaly showed by path.
     * @param dest place to add result
     * @param path path fragments, optional
     */
    virtual void getLeafsBoundingBoxesToVec(std::vector<Box>& dest, const PathHints* path = 0) const = 0;

    /**
     * Calculate bounding boxes of all leafs, optionaly showed by path.
     * @param path path fragments, optional
     * @return bounding boxes of all leafs
     */
    std::vector<Box> getLeafsBoundingBoxes(const PathHints* path = 0) const {
        std::vector<Box> result;
        getLeafsBoundingBoxesToVec(result, path);
        return result;
    }

    /**
     * Calculate bounding boxes of all leafs, optionaly showed by path.
     * @param path path fragments
     * @return bounding boxes of all leafs
     */
    std::vector<Box> getLeafsBoundingBoxes(const PathHints& path) const {
        std::vector<Box> result;
        getLeafsBoundingBoxesToVec(result, &path);
        return result;
    }

    /**
     * Get all leafs and its translations in subtree with this in root.
     * @return all leafs and its translations in subtree with this in root.
     *
     * Some leafs can have all vector of NaNs as trasnalations.
     * This mean that translation is not well defined (some space changer on path).
     */
    virtual std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > getLeafsWithTranslations() const = 0;

    //virtual getChildInfo  //translation, bounding box

};

}       // namespace plask

#endif	// PLASK__GEOMETRY_ELEMENT_H
