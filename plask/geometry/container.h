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

namespace plask {

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

    /**
     * Remove all children which fulfil predicate.
     * @param predicate return true only if child passed as argument should be deleted
     * @tparam PredicateT functor which can take child as argument and return something convertable to bool
     */
    template <typename PredicateT>
    void removeAll(PredicateT predicate) {
        children.erease(
            std::remove_if(children.begin(), children.end(), predicate),
            children.end()
        );
    }

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

    virtual void getLeafsBoundingBoxesToVec(std::vector<Rect>& dest, const PathHints* path = 0) const {
        if (path) {
            auto c = path->getTranslationChild<dim>(*this);
            if (c) {
                c->getLeafsBoundingBoxesToVec(dest, path);
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

    /**
     * Remove all children which fulfil predicate.
     * @param predicate return true only if child passed as argument should be deleted
     * @tparam PredicateT functor which can take child as argument and return something convertable to bool
     */
    template <typename PredicateT>
    void remove(PredicateT predicate) {
        removeAll(predicate);
        updateAllHeights();
    }

    /**
     * Remove all children exactly equal to @a el.
     * @param el child(ren) to remove
     */
    void remove(const ChildType* el) {
        removeAll([&el](ChildType* c) { return c->child == el; });
        updateAllHeights();
    }

    /**
     * Remove child pointed, for this container, in @a hints.
     * @param hints path hints, see @ref geometry_paths
     */
    void remove(const PathHints& hints) {
        auto c = hints.getChild(this);
        if (c) {
            children.erase(std::find(children.begin(), children.end(), c));
            updateAllHeights();
        }
    }

    protected:

    /**
     * stackHeights[x] is current stack heights with x first elements in it (sums of heights of first x elements),
     * stackHeights.size() = children.size() + 1
     */
    std::vector<double> stackHeights;

    /**
     * Calculate element up translation and height of stack with element @a el.
     * @param el[in] geometry element (typically: which is or will be in stack)
     * @param prev_height[in] height of stack under an @a el
     * @param el_translation[out] up translation which should element @a el have
     * @param next_height[out] height of stack with an @a el on top (up to @a el)
     */
    void calcHeight(const shared_ptr<ChildType>& el, double prev_height, double& el_translation, double& next_height) {
        auto bb = el->getBoundingBox();
        el_translation = prev_height - bb.lower.up;
        next_height = bb.upper.up + el_translation;
    }

    /**
     * Update stack height (fragment with pointed child on top) and pointed child up translation.
     * @param child_index index of child
     */
    void updateHeight(std::size_t child_index) {
        calcHeight(children[child_index]->child, stackHeights[child_index], children[child_index]->translation.up, children[child_index+1]);
    }

    /**
     * Update stack heights and up translation of all children, with indexes from @a first_child_index.
     * @param first_child_index index of first child to update
     */
    void updateAllHeights(std::size_t first_child_index = 0) {
        for ( ; first_child_index < children.size(); ++first_child_index)
            updateHeight(first_child_index);
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
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(const shared_ptr<ChildType>& el, const double tran_translation = 0.0);

    /**
     * Add child to stack top.
     * @param el element to add
     * @param tran_translation horizontal translation of element
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint push_back(shared_ptr<ChildType> el, const double tran_translation = 0.0) { return add(el, tran_translation); }

    /**
     * Add children to stack top.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el element to add
     * @param tran_translation horizontal translation of element
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint addUnsafe(const shared_ptr<ChildType>& el, const double tran_translation = 0.0);

    /**
     * Add children to stack bottom, move all other children higher.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el element to add
     * @param tran_translation horizontal translation of element
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint push_front_Unsafe(const shared_ptr<ChildType>& el, const double tran_translation = 0.0);

    /**
     * Add child to stack bottom, move all other children higher.
     * @param el element to add
     * @param tran_translation horizontal translation of element
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint push_front(const shared_ptr<ChildType>& el, const double tran_translation = 0.0);

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
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(const shared_ptr<ChildType>& el, const double lon_translation = 0.0, const double tran_translation = 0.0);

    /**
     * Add children to stack top.
     * @param el element to add
     * @param lon_translation, tran_translation horizontal translation of element
     * @return path hint, see @ref geometry_paths
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
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint addUnsafe(const shared_ptr<ChildType>& el, const double lon_translation = 0.0, const double tran_translation = 0.0);

    PathHints::Hint push_front_Unsafe(const shared_ptr<ChildType>& el, const double lon_translation = 0.0, const double tran_translation = 0.0);

    PathHints::Hint push_front(const shared_ptr<ChildType>& el, const double lon_translation = 0.0, const double tran_translation = 0.0);
};

template <int dim>
class MultiStackContainer: public chooseType<dim-2, StackContainer2d, StackContainer3d>::type {

    ///Type of parent class of this.
    typedef typename chooseType<dim-2, StackContainer2d, StackContainer3d>::type UpperClass;

    /*
     * @param a, divider
     * @return \f$a - \floor{a / divider} * divider\f$
     */
    /*static double modulo(double a, double divider) {
        return a - static_cast<double>( static_cast<int>( a / divider ) ) * divider;
    }*/

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
        height = std::fmod(zeroBasedRequestHeight, zeroBasedStackHeight) + stackHeights.front();
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

    virtual void getLeafsBoundingBoxesToVec(std::vector<Rect>& dest, const PathHints* path = 0) const {
        std::size_t old_size = dest.size();
        UpperClass::getLeafsBoundingBoxesToVec(dest, path);
        std::size_t new_size = dest.size();
        const double minusZeroBasedStackHeight = stackHeights.front() - stackHeights.back();
        for (unsigned r = 1; r < repeat_count; ++r) {
            dest.insert(dest.end(), dest.begin() + old_size, dest.begin() + new_size);
            for (auto i = dest.end() - (new_size-old_size); i != dest.end(); ++i)
                i->translateUp(minusZeroBasedStackHeight * r);
        }
    }

    virtual void getLeafsToVec(std::vector< shared_ptr<const GeometryElement> >& dest) const {
        std::size_t old_size = dest.size();
        UpperClass::getLeafsToVec(dest);
        std::size_t new_size = dest.size();
        for (unsigned r = 1; r < repeat_count; ++r)
            dest.insert(dest.end(), dest.begin() + old_size, dest.begin() + new_size);
    }
    
    virtual std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > getLeafsWithTranslations() const {
        std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > result = UpperClass::getLeafsWithTranslations();
        std::size_t size = result.size();   //oryginal size
        const double minusZeroBasedStackHeight = stackHeights.front() - stackHeights.back();
        for (unsigned r = 1; r < repeat_count; ++r) {
            result.insert(result.end(), result.begin(), result.begin() + size);
            for (auto i = result.end() - size; i != result.end(); ++i)
                std::get<1>(*i).up += minusZeroBasedStackHeight * r;
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
