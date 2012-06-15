#ifndef PLASK__GEOMETRY_ELEMENT_H
#define PLASK__GEOMETRY_ELEMENT_H

/** @file
This file includes base class for geometries elements.
*/


#include <vector>
#include <tuple>
#include <functional>

#include "../material/material.h"
#include "../material/air.h"
#include "primitives.h"
#include "../utils/iterators.h"

#include <boost/signals2.hpp>
#include "../utils/event.h"

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

template < int dimensions > struct GeometryElementD;

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
    struct Event: public EventWithSourceAndFlags<GeometryElement> {

        /// Event flags (which describes event properties).
        enum Flags {
            DELETE = 1,             ///< is deleted
            RESIZE = 1<<1,          ///< size could be changed
            DELEGATED = 1<<2,       ///< delegated from child
            CHILD_LIST = 1<<3,      ///< children list was changed
            USER_DEFINED = 1<<4     ///< user-defined flags could have ids: USER_DEFINED, USER_DEFINED<<1, USER_DEFINED<<2, ...
        };

        /**
         * Get event's flags for parent in tree of geometry
         * (useful to calculate flags for event which should be generated by parent of element which is a source of this event).
         * @return flags for parent in tree of geometry
         */
        unsigned char flagsForParent() const { return flagsWithout(DELETE | CHILD_LIST) | DELEGATED | RESIZE; }

        /**
         * Check if given @p flag is set.
         * @param flag flag to check
         * @return @c true only if @p flag is set
         */
        bool hasFlag(Flags flag) const { return hasAnyFlag(flag); }

        /**
         * Check if DELETE flag is set, which mean that source of event is deleted.
         * @return @c true only if DELETE flag is set
         */
        bool isDelete() const { return hasFlag(DELETE); }

        /**
         * Check if RESIZE flag is set, which mean that source of event could be resized.
         * @return @c true only if RESIZE flag is set
         */
        bool isResize() const { return hasFlag(RESIZE); }

        /**
         * Check if DELEGATED flag is set, which mean that source delegate event from its child.
         * @return @c true only if DELEGATED flag is set
         */
        bool isDelgatedFromChild() const { return hasFlag(DELEGATED); }

        /**
         * Check if CHILD_LIST flag is set, which mean that children list of source could be changed.
         * @return @c true only if CHILD_LIST flag is set
         */
        bool hasChangedChildrenList() const { return hasFlag(CHILD_LIST); }

        /**
         * Construct event.
         * @param source source of event
         * @param flags which describes event's properties
         */
        explicit Event(GeometryElement& source, unsigned char flags = 0):  EventWithSourceAndFlags<GeometryElement>(source, flags) {}
    };

    /**
     * This structure can refer to part of geometry tree.
     */
    struct Subtree {

        /// Geometry element.
        shared_ptr<const GeometryElement> element;

        /// Some (but not necessary all) children of element.
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
        bool empty() const { return !element; }
    };

    /**
     * Base class for geometry changers.
     *
     * Geometry changer can change GeometryElement to another one.
     */
    struct Changer {

        virtual ~Changer() {}

        /**
         * Try to apply changes.
         * @param to_change[in,out] pointer to element which eventualy will be changed (in such case pointer after call can point to another geometry element)
         * @param translation[out] optional, extra translation for element after change (in case of 2d object caller reads only \a tran and \a up components of this vector)
         * @return @c true only if something was changed, @c false if nothing was changed (in such case changer doesn't change arguments)
         */
        virtual bool apply(shared_ptr<const GeometryElement>& to_change, Vec<3, double>* translation = 0) const = 0;

    };

    /**
     * Geometry changer which hold vector of changers and try to apply this changers sequently.
     *
     * Its apply method call: changers[0].apply(to_change, translation), changers[1].apply(to_change, translation), ...
     * up to time when one of this call returns @c true (and then it returns @c true) or
     * there are no mora changers in changes vector (and then it returns @c false).
     */
    struct CompositeChanger: public Changer {

        std::vector<const Changer*> changers;

        /**
         * Construct CompositeChanger and append @p changer to its changers list.
         * @param changer changer to append
         */
        CompositeChanger(const Changer* changer);

        /**
         * Append @p changer to changers list.
         * @param changer changer to append
         * @return @c *this
         */
        CompositeChanger& operator()(const Changer* changer);

        /// Delete all holded changers (using delete operator).
        ~CompositeChanger();

        virtual bool apply(shared_ptr<const GeometryElement>& to_change, Vec<3, double>* translation = 0) const;

    };

    /**
     * Changer which replaces given geometry element @a from to given geometry element @a to.
     */
    struct ReplaceChanger: public Changer {

        shared_ptr<const GeometryElement> from, to;

        /// Translation to return by apply.
        Vec<3, double> translation;

        /// Construct uninitilized changer.
        ReplaceChanger() {}

        /**
         * Construct changer which change @p from to @p to and return given @p translation.
         * @param from, to, translation changer parameters
         */
        ReplaceChanger(const shared_ptr<const GeometryElement>& from, const shared_ptr<const GeometryElement>& to, Vec<3, double> translation)
            : from(from), to(to), translation(translation) {}

        /**
         * Construct changer which change @p from to calc_replace(to) and return zeroed translation.
         * @param from element which should be changed
         * @param functor which is used to calculate change destination element
         */
        template <typename F>
        ReplaceChanger(const shared_ptr<const GeometryElement>& from, F calc_replace): from(from), translation(0.0, 0.0, 0.0) {
            this->to = calc_replace(this->from);
        }

        virtual bool apply(shared_ptr<const GeometryElement>& to_change, Vec<3, double>* translation = 0) const;

    };

    /**
     * Changer which replaces given geometry element @a toChange to block (2d or 3d, depents from @a toChange)
     * with size equals to @a toChange bounding box, and with given material.
     */
    struct ToBlockChanger: public ReplaceChanger {

        ToBlockChanger(const shared_ptr<const GeometryElement>& toChange, const shared_ptr<Material>& material);

    };

    /// Predicate on GeometryElement
    typedef std::function<bool(const GeometryElement&)> Predicate;

    /// Predicate which check if given element is leaf.
    static bool PredicateIsLeaf(const GeometryElement& el) { return el.isLeaf(); }

    /// Predicate which check if given element is identical to other, holded element (given in constructor).
    struct PredicateIsIdenticalTo {
        const GeometryElement& elementToBeEqual;
        PredicateIsIdenticalTo(const GeometryElement& elementToBeEqual): elementToBeEqual(elementToBeEqual) {}
        PredicateIsIdenticalTo(const shared_ptr<GeometryElement>& elementToBeEqual): elementToBeEqual(*elementToBeEqual) {}
        PredicateIsIdenticalTo(const shared_ptr<const GeometryElement>& elementToBeEqual): elementToBeEqual(*elementToBeEqual) {}
        bool operator()(const GeometryElement& el) const { return &el == &elementToBeEqual; }
    };

    /// Changed signal, fired when element was changed.
    boost::signals2::signal<void(const Event&)> changed;

    /// Connect a method to changed signal
    template <typename ClassT, typename methodT>
    boost::signals2::connection changedConnectMethod(ClassT* obj, methodT method) {
        return changed.connect(boost::bind(method, obj, _1));
    }

    /// Disconnect a method from changed signal
    template <typename ClassT, typename methodT>
    void changedDisconnectMethod(ClassT* obj, methodT method) {
        changed.disconnect(boost::bind(method, obj, _1));
    }

    /**
     * Call changed with this as event source.
     * @param event_constructor_params_without_source parameters for event constructor (without first - source)
     */
    template<typename EventT = Event, typename ...Args>
    void fireChanged(Args&&... event_constructor_params_without_source) {
        changed(EventT(*this, std::forward<Args>(event_constructor_params_without_source)...));
    }

    /**
     * Initialize this to be the same as @p to_copy but doesn't have any changes observer.
     * @param to_copy object to copy
     */
    GeometryElement(const GeometryElement& to_copy) {}

    /**
     * Set this to be the same as @p to_copy but doesn't changed changes observer.
     * @param to_copy object to copy
     */
    GeometryElement& operator=(const GeometryElement& to_copy) { return *this; }

    GeometryElement() = default;

    //TODO

    /**
     * Virtual destructor. Inform all change listeners.
     */
    virtual ~GeometryElement();

    /**
     * Cast this to GeometryElementD<DIMS>.
     * @return this casted to GeometryElementD<DIMS> or nullptr if casting is not possible.
     */
    template<int DIMS>
    shared_ptr< GeometryElementD<DIMS> > asD();

    /**
     * Cast this to GeometryElementD<DIMS> (const version).
     * @return this casted to GeometryElementD<DIMS> or nullptr if casting is not possible.
     */
    template<int DIMS>
    shared_ptr< const GeometryElementD<DIMS> > asD() const;

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
    //TODO predicate, path
    virtual bool isInSubtree(const GeometryElement& el) const = 0;

    /**
     * Find paths to @a el.
     * @param el element to search for
     * @param pathHints (optional) path hints which limits search space
     * @return sub-tree with paths to given element (@p el is in all leafs), empty sub-tree if @p el is not in subtree with @c this in root
     */
    //TODO predicate
    virtual Subtree findPathsTo(const GeometryElement& el, const PathHints* pathHints = 0) const = 0;

    /**
     * Append all elements from subtree with this in root, which fullfil predicate to vector @p dest.
     * @param predicate
     * @param dest destination vector
     * @param path (optional) path hints which limits search space
     */
    virtual void getElementsToVec(const Predicate& predicate, std::vector< shared_ptr<const GeometryElement> >& dest, const PathHints* path = 0) const = 0;

    /**
     * Append all elements from subtree with this in root, which fullfil predicate to vector @p dest.
     * @param predicate
     * @param dest destination vector
     * @param path path hints which limits search space
     */
    void getElementsToVec(const Predicate& predicate, std::vector< shared_ptr<const GeometryElement> >& dest, const PathHints& path) const {
        getElementsToVec(predicate, dest, &path);
    }

    /**
     * Append all leafs in subtree with this in root to vector @p dest.
     * @param dest leafs destination vector
     * @param path (optional) path hints which limits search space
     */
    void getLeafsToVec(std::vector< shared_ptr<const GeometryElement> >& dest, const PathHints* path = 0) const {
        getElementsToVec(&GeometryElement::PredicateIsLeaf, dest, path);
    }

    /**
     * Append all leafs in subtree with this in root to vector @p dest.
     * @param dest leafs destination vector
     * @param path path hints which limits search space
     */
    void getLeafsToVec(std::vector< shared_ptr<const GeometryElement> >& dest, const PathHints& path) const {
        getLeafsToVec(dest, &path);
    }

    /**
     * Get all leafs in subtree with this object as root.
     * @param path (optional) path hints which limits search space
     * @return all leafs in subtree with this object as root
     */
    std::vector< shared_ptr<const GeometryElement> > getLeafs(const PathHints* path = 0) const {
        std::vector< shared_ptr<const GeometryElement> > result;
        getLeafsToVec(result, path);
        return result;
    }

    /**
     * Get all leafs in subtree with this object as root.
     * @param path path hints which limits search space
     * @return all leafs in subtree with this object as root
     */
    std::vector< shared_ptr<const GeometryElement> > getLeafs(const PathHints& path) const {
        return getLeafs(&path);
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
    virtual std::size_t getRealChildrenCount() const;

    /**
     * Get real (physicaly stored) child with given index.
     *
     * By default call getChildAt(child_nr), but elements of some types (like multi-stack) redefine this.
     * @param child_nr index of real child to get
     * @return child with index @p child_nr
     */
    virtual shared_ptr<GeometryElement> getRealChildAt(std::size_t child_nr) const;

    /**
     * Remove child at given @p index.
     *
     * This is unsafe but fast version, it doesn't check index and doesn't call fireChildrenChanged() to inform listeners about this object changes.
     * Caller should do this manually or call removeAt(std::size_t) instead.
     *
     * Default implementation throw excption but this method is overwritten in subclasses.
     * @param index index of real child to remove
     */
    virtual void removeAtUnsafe(std::size_t index);

    /**
     * Remove child at given @p index.
     *
     * Throw exception if given @p index is not valid, real child index.
     * @param index index of real child to remove
     */
    void removeAt(std::size_t index) {
        ensureIsValidChildNr(index, "removeAt", "index");
        removeAtUnsafe(index);
        fireChildrenChanged();
    }

    void removeRangeUnsafe(std::size_t index_begin, std::size_t index_end) {
        while (index_begin < index_end) removeAtUnsafe(--index_end);
    }

    /**
     * Remove all children in given range [index_begin, index_end).
     * @param index_begin, index_end range of real children's indexes
     * @return true if something was delete
     */
    bool removeRange(std::size_t index_begin, std::size_t index_end) {
        if (index_begin >= index_end) return false;
        ensureIsValidChildNr(index_end-1, "removeRange", "index_end-1");
        removeRangeUnsafe(index_begin, index_end);
        fireChildrenChanged();
        return true;
    }

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

    bool canHasAsChild(const GeometryElement& potential_child) const { return !potential_child.isInSubtree(*this); }
    bool canHasAsParent(const GeometryElement& potential_parent) const { return !this->isInSubtree(potential_parent); }

    /**
     * Throw CyclicReferenceException if @p potential_parent is in subtree with this in root.
     * @param potential_parent[in] potential, new parent of this
     */
    void ensureCanHasAsParent(const GeometryElement& potential_parent) const;

    /**
     * Throw CyclicReferenceException if @p potential_child has this in subtree.
     * @param potential_child[in] potential, new child of this
     */
    void ensureCanHaveAsChild(const GeometryElement& potential_child) const { potential_child.ensureCanHasAsParent(*this); }

protected:

    /**
     * Check if given @p index is valid child index and throw exception of it is not.
     * @param child_nr index to check
     * @param method_name caller method name which is used to format excption message
     * @param arg_name name of index argument in caller method, used to format excption message
     * @throw OutOfBoundException if index is not valid
     */
    void ensureIsValidChildNr(std::size_t child_nr, const char* method_name = "getChildAt", const char* arg_name = "child_nr") const {
        std::size_t children_count = getRealChildrenCount();
        if (child_nr >= children_count)
            throw OutOfBoundException(method_name, arg_name, child_nr, 0, children_count-1);
    }

    /// Inform observers that children list was changed (also that this is resized)
    void fireChildrenChanged() {
        this->fireChanged(GeometryElement::Event::RESIZE | GeometryElement::Event::CHILD_LIST);
    }

};

template <int dim> struct Translation;

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
     * Calculate minimal rectangle which includes all points of real geometry element.
     *
     * This box can be diffrent from getBoundingBox() only for elements which have virtual children, like multple-stack.
     * Returned box is always included in (in most cases: equal to) box returned by getBoundingBox().
     *
     * Default implementation returns result of getBoundingBox() call.
     * @return calculated rectangle
     */
    virtual Box getRealBoundingBox() const { return getBoundingBox(); }

    /**
     * Return material in a given point inside the geometry element
     * @param p point
     * @return material in given point, or @c nullptr if this GeometryElement not includes point @a p
     */
    virtual shared_ptr<Material> getMaterial(const DVec& p) const = 0;

    /**
     * Return material in a given point inside the geometry element
     * @param p point
     * @return material in given point, or Air if this GeometryElement not includes point @a p
     */
    shared_ptr<Material> getMaterialOrAir(const DVec& p) const {
        auto real_mat = getMaterial(p);
        return real_mat ? real_mat : make_shared<Air>();
    }

    /**
     * Calculate and append to vector bounding boxes of all nodes which fulfill given @p predicate, optionally marked by path.
     * @param predicate
     * @param dest place to add result, bounding boxes will be added in the same order which is generated by GeometryElement::getElements
     * @param path path fragments, optional
     */
    virtual void getBoundingBoxesToVec(const GeometryElement::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const = 0;

    /**
     * Calculate and append to vector bounding boxes of all nodes which fulfill given @p predicate, marked by path.
     * @param predicate
     * @param dest place to add result, bounding boxes will be added in the same order which is generated by GeometryElement::getElements
     * @param path path fragments
     */
    void getBoundingBoxesToVec(const GeometryElement::Predicate& predicate, std::vector<Box>& dest, const PathHints& path) const {
        getBoundingBoxesToVec(predicate, dest, &path);
    }

    /**
     * Calculate the vector of bounding boxes of all nodes which fulfill given @p predicate, optionally marked by path.
     * @param predicate
     * @param path path fragments, optional
     * @return vector of bounding boxes of all nodes which fulfill given @p predicate, optionally marked by path
     */
    std::vector<Box> getBoundingBoxes(const GeometryElement::Predicate& predicate, const PathHints* path = 0) const {
        std::vector<Box> result;
        getBoundingBoxesToVec(predicate, result, path);
        return result;
    }

    /**
     * Calculate the vector of bounding boxes of all nodes which fulfill given @p predicate, marked by path.
     * @param predicate
     * @param path path fragments
     * @return vector of bounding boxes of all nodes which fulfill given @p predicate, optionally marked by path
     */
    std::vector<Box> getBoundingBoxes(const GeometryElement::Predicate& predicate, const PathHints& path) {
        return getBoundingBoxes(predicate, &path);
    }

    /**
     * Calculate and append to vector bounding boxes of all leafs, optionally marked by path.
     * @param dest place to add result, bounding boxes will be added in the same order which is generated by GeometryElement::getLeafsToVec
     * @param path path fragments, optional
     */
    void getLeafsBoundingBoxesToVec(std::vector<Box>& dest, const PathHints* path = 0) const {
        getBoundingBoxesToVec(&GeometryElement::PredicateIsLeaf, dest, path);
    }

    /**
     * Calculate and append to vector bounding boxes of all leafs, marked by path.
     * @param dest place to add result, bounding boxes will be added in the same order which is generated by GeometryElement::getLeafsToVec
     * @param path path fragments
     */
    void getLeafsBoundingBoxesToVec(std::vector<Box>& dest, const PathHints& path) const {
        getLeafsBoundingBoxesToVec(dest, &path);
    }

    /**
     * Calculate bounding boxes of all leafs, optionally marked by path.
     * @param path path fragments, optional
     * @return bounding boxes of all leafs, in the same order which is generated by GeometryElement::getLeafs(const PathHints*)
     */
    std::vector<Box> getLeafsBoundingBoxes(const PathHints* path = 0) const {
        std::vector<Box> result;
        getLeafsBoundingBoxesToVec(result, path);
        return result;
    }

    /**
     * Calculate bounding boxes of all leafs, marked by path.
     * @param path path fragments
     * @return bounding boxes of all leafs, in the same order which is generated by GeometryElement::getLeafs(const PathHints&)
     */
    std::vector<Box> getLeafsBoundingBoxes(const PathHints& path) const {
        return getLeafsBoundingBoxes(&path);
    }

    /**
     * Calculate and append to vector positions of all nodes which fulfill given @p predicate, optionally marked by path.
     *
     * Some elements can have all vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     * @param predicate
     * @param dest place to add result, positions will be added in the same order which is generated by GeometryElement::getElementsToVec
     * @param path path fragments, optional
     */
    virtual void getPositionsToVec(const Predicate& predicate, std::vector<DVec>& dest, const PathHints* path = 0) const = 0;

    /**
     * Calculate and append to vector positions of all nodes which fulfill given @p predicate, marked by path.
     *
     * Some elements can have all vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     * @param predicate
     * @param dest place to add result, positions will be added in the same order which is generated by GeometryElement::getElementsToVec
     * @param path path fragments
     */
    void getPositionsToVec(const Predicate& predicate, std::vector<DVec>& dest, const PathHints& path) const {
        getPositionsToVec(predicate, dest, &path);
    }

    /**
     * Calculate and append to vector positions of all nodes which fulfill given @p predicate, optionally marked by path.
     * @param predicate
     * @param path path fragments, optional
     * @return positions of the pointed elements in the sub-tree with this element in the root, in the same order which is generated by GeometryElement::getElements
     *
     * Some elements can have all vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     */
    std::vector<DVec> getPositions(const Predicate& predicate, const PathHints* path = 0) const {
        std::vector<DVec> result;
        getPositionsToVec(predicate, result, path);
        return result;
    }

    /**
     * Calculate and append to vector positions of all nodes which fulfill given @p predicate, marked by path.
     * @param predicate
     * @param path path fragments
     * @return positions of the pointed elements in the sub-tree with this element in the root, in the same order which is generated by GeometryElement::getElements
     *
     * Some elements can have all vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     */
    std::vector<DVec> getPositions(const Predicate& predicate, const PathHints& path) const {
        return getPositions(predicate, &path);
    }

    /**
     * Calculate and append to vector positions of all leafs, optionally marked by path.
     *
     * Some leafs can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     * @param predicate
     * @param dest place to add result, positions will be added in the same order which is generated by GeometryElement::getLeafsToVec
     * @param path path fragments, optional
     */
    void getLeafsPositionsToVec(std::vector<DVec>& dest, const PathHints* path = 0) const {
        getPositionsToVec(&PredicateIsLeaf, dest, path);
    }

    /**
     * Calculate and append to vector positions of all leafs, marked by path.
     *
     * Some leafs can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     * @param predicate
     * @param dest place to add result, positions will be added in the same order which is generated by GeometryElement::getLeafsToVec
     * @param path path fragments
     */
    void getLeafsPositionsToVec(std::vector<DVec>& dest, const PathHints& path) const {
        getLeafsPositionsToVec(dest, &path);
    }

    /**
     * Calculate and return a vector of positions of all leafs, optionally marked by path.
     * @param predicate
     * @param path path fragments, optional
     * @return positions of leafs in the sub-tree with this element in the root, in the same order which is generated by GeometryElement::getLeafs
     *
     * Some leafs can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     */
    std::vector<DVec> getLeafsPositions(const PathHints* path = 0) const {
        std::vector<DVec> result;
        getLeafsPositionsToVec(result, path);
        return result;
    }

    /**
     * Calculate and return a vector of positions of all leafs, marked by path.
     * @param predicate
     * @param path path fragments
     * @return positions of leafs in the sub-tree with this element in the root, in the same order which is generated by GeometryElement::getLeafs
     *
     * Some leafs can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     */
    std::vector<DVec> getLeafsPositions(const PathHints& path) const {
        return getLeafsPositions(&path);
    }

    /**
     * Calculate and append to vector positions of all instances of given @p element, optionally marked by path.
     *
     * Some elements can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     * @param dest place to add result, positions will be added in the same order which is generated by GeometryElement::getElementsToVec
     * @param element element to which instances translations should be found
     * @param path path fragments, optional
     */
    void getElementPositionToVec(std::vector<DVec>& dest, const GeometryElement& element, const PathHints* path = 0) const {
        getPositionsToVec(PredicateIsIdenticalTo(element), dest, path);
    }

    /**
     * Calculate and append to vector positions of all instances of given @p element, marked by path.
     *
     * Some elements can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     * @param dest place to add result, positions will be added in the same order which is generated by GeometryElement::getElementsToVec
     * @param element element to which instances translations should be found
     * @param path path fragments
     */
    void getElementPositionToVec(std::vector<DVec>& dest, const GeometryElement& element, const PathHints& path) const {
        getElementPositionToVec(dest, element, &path);
    }

    /**
     * Calculate and return a vector of positions of all instances of given @p element, optionally marked by path.
     * @param element element to which instances translations should be found
     * @param path path fragments, optional
     * @return vector of positions
     *
     * Some elements can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     */
    std::vector<DVec> getElementPositions(const GeometryElement& element, const PathHints* path = 0) const {
        return getPositions(PredicateIsIdenticalTo(element), path);
    }

    /**
     * Calculate and return a vector of positions of all instances of given @p element, marked by path.
     * @param element element to which instances translations should be found
     * @param path path fragments
     * @return vector of positions
     *
     * Some elements can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     */
    std::vector<DVec> getElementPositions(const GeometryElement& element, const PathHints& path) const {
        return getElementPositions(element, &path);
    }

    /**
     * Get @p element wrapped with translation to be in corrdinate space of this.
     *
     * @param element element which should be in subtree of this
     * @param path path fragments, optional
     * @return @p element wrapped with translation or shared_ptr< Translation<dimensions> >() if translation os not well-defined
     */
    shared_ptr< Translation<dimensions> > getElementInThisCordinates(const shared_ptr< GeometryElementD<dimensions> >& element, const PathHints* path = 0) const;

    /**
     * Get @p element wrapped with translation to be in corrdinate space of this.
     *
     * @param element element which should be in subtree of this
     * @param path path fragments
     * @return @p element wrapped with translation or shared_ptr< Translation<dimensions> >() if translation os not well-defined
     */
    shared_ptr< Translation<dimensions> > getElementInThisCordinates(const shared_ptr< GeometryElementD<dimensions> >& element, const PathHints& path) const {
        return getElementInThisCordinates(element, &path);
    }

    /**
     * Get @p element wrapped with translation to be in corrdinate space of this.
     *
     * Throw excpetion if element doesn't have well-defined, one translation.
     * @param element element which should be in subtree of this
     * @param path path fragments, optional
     * @return @p element wrapped with translation
     */
    shared_ptr< Translation<dimensions> > requireElementInThisCordinates(const shared_ptr< GeometryElementD<dimensions> >& element, const PathHints* path = 0) const {
        shared_ptr< Translation<dimensions> > result = getElementInThisCordinates(element, path);
        if (!result) throw Exception("Translation to element required in local coordinates is not well defined");
        return result;
    }

    /**
     * Get @p element wrapped with translation to be in corrdinate space of this.
     *
     * Throw excpetion if element doesn't have well-defined, one translation.
     * @param element element which should be in subtree of this
     * @param path path fragments
     * @return @p element wrapped with translation
     */
    shared_ptr< Translation<dimensions> > requireElementInThisCordinates(const shared_ptr< GeometryElementD<dimensions> >& element, const PathHints& path) const {
        return requireElementInThisCordinates(element, &path);
    }

};

} // namespace plask

#endif // PLASK__GEOMETRY_ELEMENT_H







