#ifndef PLASK__GEOMETRY_CONTAINER_H
#define PLASK__GEOMETRY_CONTAINER_H

/** @file
This file includes containers of geometries elements.
*/

#include <map>
#include <vector>
#include <algorithm>
#include "element.h"
#include "transform.h"
#include "../utils/metaprog.h"

namespace plask {

/**
Represent hints for path finder.

Hints are used to to find unique path for all GeometryElement pairs,
even if one of the pair element is inserted to geometry graph in more than one place.

Each hint allow to choose one child for geometry element container and it is a pair:
geometry element container -> element in container.

Typically, hints are returned by methods which adds new elements to containers.
*/
struct PathHints {

    ///Type for map: geometry element container -> element in container
    typedef std::map< weak_ptr<GeometryElement>, weak_ptr<GeometryElement> > HintMap;

    ///Pair type: geometry element container -> element in container
    typedef HintMap::value_type Hint;

    ///Hints map.
    HintMap hintFor;

    /**
     * Add hint to hints map. Overwrite if hint for given container already exists.
     * @param hint hint to add
     */
    void addHint(const Hint& hint);

    /**
     * Add hint to hints map. Overwrite if hint for given container already exists.
     * @param hint hint to add
     */
    PathHints& operator+=(const Hint& hint) {
        addHint(hint);
        return *this;
    }

    /**
     * Add hint to hints map. Overwrite if hint for given container already exists.
     * @param container, child hint to add
     */
    void addHint(weak_ptr<GeometryElement> container, weak_ptr<GeometryElement> child);

    /**
     * Get child for given container.
     * @return child for given container or @c nullptr if there is no hint for given container
     */
    shared_ptr<GeometryElement> getChild(shared_ptr<GeometryElement> container);
    
    /**
     * Get child for given container.
     * @return child for given container or @c nullptr if there is no hint for given container
     */
    shared_ptr<GeometryElement> getChild(shared_ptr<GeometryElement> container) const;
    
    /**
     * Remove all hints which refer to deleted objects.
     */
    void cleanDeleted();

};

/**
 * Template which implement container using stl container of pointers to some geometry elements objects (Translation by default).
 * @tparam dim GeometryElementContainer dimension
 * @tparam container_type container of pointers to children
 */
template <int dim, typename container_type = std::vector< shared_ptr< Translation<dim> > > >
struct GeometryElementContainerImpl: public GeometryElementContainer<dim> {

    ///Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::DVec DVec;

    ///Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::Rect Rect;

protected:
    container_type children;

public:

    virtual bool inside(const DVec& p) const {
        for (auto child: children) if (child->inside(p)) return true;
        return false;
    }

    virtual bool intersect(const Rect& area) const {
        for (auto child: children) if (child->intersect(area)) return true;
        return false;
    }

    virtual Rect getBoundingBox() const {
        //if (childs.empty()) throw?
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

    virtual std::vector<Rect> getLeafsBoundingBoxes() const {
        std::vector<Rect> result;
        for (auto child: children) {
            std::vector<Rect> child_leafs_boxes = child->getLeafsBoundingBoxes();
            result.insert(result.end(), child_leafs_boxes.begin(), child_leafs_boxes.end());
        }
        return result;
    }

    virtual bool isInSubtree(GeometryElement& el) const {
        if (&el == this) return true;
        for (auto child: children)
            if (child->isInSubtree(el))
                return true;
        return false;
    }
};

/**
 * Geometry elements container in which every child has an associated translation vector.
 */
//TODO some implementation are naive, and can be done faster with some caches
template < int dim >
struct TranslationContainer: public GeometryElementContainerImpl<dim> {

    ///Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::DVec DVec;

    ///Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::Rect Rect;

    ///Type of this child.
    typedef GeometryElementD<dim> ChildType;

    ///Type of translation geometry elment in space of this.
    typedef Translation<dim> TranslationT;

    using GeometryElementContainerImpl<dim>::children;
    using GeometryElementContainerImpl<dim>::shared_from_this;

    /**
     * Add new child (translated) to end of children vector.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el new child
     * @param translation trasnalation of child
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
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(const shared_ptr<ChildType>& el, const DVec& translation = Primitive<dim>::ZERO_VEC) {
        ensureCanHasAsChild(*el);
        return addUnsafe(el, translation);
    }
    
    /**
     * Remove all children exactly equal to @a el.
     * @param el child(ren) to remove
     */
    void remove(const ChildType* el) {
        children.erease(
            std::remove_if(children.begin(), children.end(), [&el](ChildType* c) { return c->child == el; }),
            children.end()
        );
    }
    
    /**
     * Remove child pointed, for this container, in @a hints.
     * @param hints path hints
     */
    void remove(const PathHints& hints) {
        auto c = hints.getChild(this);
        if (c) children.erase(std::find(children.begin(), children.end(), c));
    }

};

/**
 * Common code for stack containers (which have children in stack/layers).
 */
template <int dim>
struct StackContainerBaseImpl: public GeometryElementContainerImpl<dim> {

    ///Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::DVec DVec;

    ///Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::Rect Rect;

    ///Type of this child.
    typedef GeometryElementD<dim> ChildType;

    ///Type of translation geometry elment in space of this.
    typedef Translation<dim> TranslationT;

    using GeometryElementContainerImpl<dim>::children;

    /**
     * @param baseHeight height where should start first element
     */
    explicit StackContainerBaseImpl(const double baseHeight = 0.0) {
        stackHeights.push_back(baseHeight);
    }

    /**
     * @param height
     * @return child which are on given @a height or @c nullptr
     */
    const TranslationT* getChildForHeight(double height) const {
        auto it = std::lower_bound(stackHeights.begin(), stackHeights.end(), height);
        if (it == stackHeights.end() || it == stackHeights.begin()) return nullptr;
        return children[it-stackHeights.begin()-1].get();
    }

    virtual bool inside(const DVec& p) const {
        const TranslationT* c = getChildForHeight(p.c1);
        return c ? c->inside(p) : false;
    }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        const TranslationT* c = getChildForHeight(p.c1);
        return c ? c->getMaterial(p) : shared_ptr<Material>();
    }

    protected:

    ///stackHeights[x] is current stack heights with x first elements in it (sums of heights of first x elements)
    std::vector<double> stackHeights;
    
    void updateHeight(std::size_t child_index) {
        
    }

};


/**
 * 2d container which have children in stack/layers.
 */
struct StackContainer2d: public StackContainerBaseImpl<2> {

    using StackContainerBaseImpl<2>::children;

    /**
     * @param baseHeight height where the first element should start
     */
    explicit StackContainer2d(const double baseHeight = 0.0);

    /**
     * Add children to stack top.
     * @param el element to add
     * @param tran_translation horizontal translation of element
     * @return path hint
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(const shared_ptr<ChildType>& el, const double tran_translation = 0.0);

    /**
     * Add child to stack top.
     * @param el element to add
     * @param tran_translation horizontal translation of element
     * @return path hint
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint push_back(shared_ptr<ChildType> el, const double tran_translation = 0.0) { return add(el, tran_translation); }

    /**
     * Add children to stack top.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el element to add
     * @param tran_translation horizontal translation of element
     * @return path hint
     */
    PathHints::Hint addUnsafe(const shared_ptr<ChildType>& el, const double tran_translation = 0.0);

};

/**
 * 3d container which have children in stack/layers.
 */
struct StackContainer3d: public StackContainerBaseImpl<3> {

    using StackContainerBaseImpl<3>::children;

    /**
     * @param baseHeight height where the first element should start
     */
    explicit StackContainer3d(const double baseHeight = 0.0);

    /**
     * Add children to stack top.
     * @param el element to add
     * @param tran_translation horizontal translation of element
     * @return path hint
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(const shared_ptr<ChildType>& el, const double lon_translation = 0.0, const double tran_translation = 0.0);

    /**
     * Add children to stack top.
     * @param el element to add
     * @param lon_translation, tran_translation horizontal translation of element
     * @return path hint
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint push_back(const shared_ptr<ChildType>& el, const double lon_translation = 0.0, const double tran_translation = 0.0) {
        return add(el, lon_translation, tran_translation);
    }

    /**
     * Add children to stack top.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el element to add
     * @param lon_translation, tran_translation horizontal translation of element
     * @return path hint
     */
    PathHints::Hint addUnsafe(const shared_ptr<ChildType>& el, const double lon_translation = 0.0, const double tran_translation = 0.0);
};

template <int dim>
class MultiStackContainer: public chooseType<dim-2, StackContainer2d, StackContainer3d>::type {

    ///Type of parent class of this.
    typedef typename chooseType<dim-2, StackContainer2d, StackContainer3d>::type UpperClass;

    /**
     * @param a, divider
     * @return \f$a - \floor{a / divider} * divider\f$
     */
    static double modulo(double a, double divider) {
        return a - static_cast<double>( static_cast<int>( a / divider ) ) * divider;
    }

public:
    using UpperClass::getChildForHeight;
    using UpperClass::stackHeights;
    using UpperClass::children;

    ///Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename UpperClass::DVec DVec;

    ///Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename UpperClass::Rect Rect;

protected:

    /*
     * Get number of all children.
     * @return number of all children
     */
    //std::size_t size() const { return children.size() * repeat_count; }

    /*
     * Get child with translation.
     * @param index index of child
     * @return child with given index
     */
    //typename UpperClass::TranslationT& operator[] (std::size_t index) { return children[index % children.size()]; }

    /**
     * Reduce @a height to the first repetition.
     * @param height to reduce
     * @return @c true only if height is inside this stack (only in such case @a height is reduced)
     */
    const bool reduceHeight(double& height) const {
        const double zeroBasedStackHeight = stackHeights.back() - stackHeights.front();
        const double zeroBasedRequestHeight = height - stackHeights.front();
        if (zeroBasedRequestHeight < 0.0 || zeroBasedRequestHeight > zeroBasedStackHeight * repeat_count)
            return false;
        height = modulo(zeroBasedRequestHeight, zeroBasedStackHeight) + stackHeights.front();
        return true;
    }

public:

    /// How many times all stack is repeated.
    unsigned repeat_count;

    /**
     * @param repeat_count how many times stack should be repeated, must be 1 or more
     * @param baseHeight height where the first element should start
     */
    explicit MultiStackContainer(unsigned repeat_count = 1, const double baseHeight = 0.0): UpperClass(baseHeight), repeat_count(repeat_count) {}

    //this is not used but, just for case redefine UpperClass::getChildForHeight
    const typename UpperClass::TranslationT* getChildForHeight(double height) const {
        if (!reduceHeight(height)) return nullptr;
        return UpperClass::getChildForHeight(height);
    }

    virtual bool intersect(const Rect& area) const {
        const double minusZeroBasedStackHeight = stackHeights.front() - stackHeights.back();
        for (unsigned r = 0; r < repeat_count; ++r)
            if (UpperClass::intersect(area.translatedUp(minusZeroBasedStackHeight*r)))
                return true;
        return false;
    }

    virtual Rect getBoundingBox() const {
        Rect result = UpperClass::getBoundingBox();
        result.upper.up += result.sizeUp() * (repeat_count-1);
        return result;
    }

    virtual std::vector<Rect> getLeafsBoundingBoxes() const {
        std::vector<Rect> result = UpperClass::getLeafsBoundingBoxes();
        std::size_t size = result.size();   //oryginal size
        const double minusZeroBasedStackHeight = stackHeights.front() - stackHeights.back();
        for (unsigned r = 1; r < repeat_count; ++r) {
            result.insert(result.end(), result.begin(), result.begin() + size);
            const double delta = minusZeroBasedStackHeight * r;
            for (auto i = result.end() - size; i != result.end(); ++i)
                i->translateUp(delta);
        }
        return result;
    }

    virtual bool inside(const DVec& p) const {
        DVec p_reduced = p;
        if (!reduceHeight(p_reduced.up)) return false;
        return UpperClass::inside(p_reduced);
    }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        DVec p_reduced = p;
        if (!reduceHeight(p_reduced.up)) return shared_ptr<Material>();
        return UpperClass::getMaterial(p_reduced);
    }

};



}	// namespace plask

#endif // PLASK__GEOMETRY_CONTAINER_H
