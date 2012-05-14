#ifndef PLASK__GEOMETRY_STACK_H
#define PLASK__GEOMETRY_STACK_H

#include "primitives.h"
#include "container.h"
#include <deque>

namespace plask {

/**
 * Common code for stack containers (which have children in stack/layers).
 * @tparam dim number of space dimentions
 * @tparam growingDirection direction in which stack growing
 */
template <int dim, int growingDirection = Primitive<dim>::DIRECTION_UP>
struct StackContainerBaseImpl: public GeometryElementContainer<dim> {

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryElementContainer<dim>::Box Box;

    /// Type of this child.
    typedef typename GeometryElementContainer<dim>::ChildType ChildType;

    /// Type of translation geometry elment in space of this.
    typedef typename GeometryElementContainer<dim>::TranslationT TranslationT;

    /// Type of the vector holiding container children
    typedef typename GeometryElementContainer<dim>::TranslationVector TranslationVector;

    using GeometryElementContainer<dim>::children;

    /**
     * @param baseHeight height where should start first element
     */
    explicit StackContainerBaseImpl(const double baseHeight = 0.0) {
        stackHeights.push_back(baseHeight);
    }

    /**
     * Get component of position in growing direction where stack starts.
     * @return component of position in growing direction where stack starts
     */
    double getBaseHeight() const { return stackHeights.front(); }

    /**
     * Set height where should start first element. Call changed.
     */
    void setBaseHeight(double newBaseHeight);

    /**
     * @param height
     * @return child which are on given @a height or @c nullptr
     */
    const shared_ptr<TranslationT> getChildForHeight(double height) const;

    virtual bool inside(const DVec& p) const {
        const shared_ptr<TranslationT> c = getChildForHeight(p.components[growingDirection]);
        return c ? c->inside(p) : false;
    }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        const shared_ptr<TranslationT> c = getChildForHeight(p.components[growingDirection]);
        return c ? c->getMaterial(p) : shared_ptr<Material>();
    }

    virtual bool removeIfTUnsafe(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate);

    virtual void removeAtUnsafe(std::size_t index);

    /// Called by child.change signal, update heights call this change
    void onChildChanged(const GeometryElement::Event& evt) {
        if (evt.isResize()) updateAllHeights(); //TODO optimization: find evt source index and update size from this index to back
        this->fireChanged(evt.flagsForParent());
    }

  protected:

    /**
     * stackHeights[x] is current stack heights with x first elements in it (sums of heights of first x elements),
     * stackHeights.size() = children.size() + 1 and stackHeights[0] is a base height (typically 0.0)
     */
    std::vector<double> stackHeights;

    /**
     * Calculate element up translation and height of stack with element @a el.
     * @param el[in] bounding box of geometry element (typically: for element which is or will be in stack)
     * @param prev_height[in] height of stack under an @a el
     * @param el_translation[out] up translation which should element @a el have
     * @param next_height[out] height of stack with an @a el on top (up to @a el)
     */
    void calcHeight(const Box& elBoudingBox, double prev_height, double& el_translation, double& next_height) {
        el_translation = prev_height - elBoudingBox.lower.components[growingDirection];
        next_height = elBoudingBox.upper.components[growingDirection] + el_translation;
    }

    /**
     * Calculate element up translation and height of stack with element @a el.
     * @param el[in] geometry element (typically: which is or will be in stack)
     * @param prev_height[in] height of stack under an @a el
     * @param el_translation[out] up translation which should element @a el have
     * @param next_height[out] height of stack with an @a el on top (up to @a el)
     */
    void calcHeight(const shared_ptr<ChildType>& el, double prev_height, double& el_translation, double& next_height) {
        calcHeight(el->getBoundingBox(), prev_height, el_translation, next_height);
    }

    /**
     * Update stack height (fragment with pointed child on top) and pointed child up translation.
     * @param child_index index of child
     */
    void updateHeight(std::size_t child_index) {
        calcHeight(children[child_index]->getChild(),
                   stackHeights[child_index],
                   children[child_index]->translation.components[growingDirection],
                   stackHeights[child_index+1]);
    }

    /**
     * Update stack heights and translation in stack growing direction of all children, with indexes from @a first_child_index.
     * @param first_child_index index of first child for which stackHeights should be update
     */
    void updateAllHeights(std::size_t first_child_index = 0) {
        for ( ; first_child_index < children.size(); ++first_child_index)
            updateHeight(first_child_index);
    }

    /**
     * Resize stackHeights (to be compatibile with children vector) and refresh its value from given index.
     * @param first_child_index index of first child for which stackHeights should be update
     */
    void rebuildStackHeights(std::size_t first_child_index = 0) {
        stackHeights.resize(children.size() + 1);
        updateAllHeights(first_child_index);
    }

};

/**
 * Horizontal stack.
 */
struct HorizontalStack: public StackContainerBaseImpl<2, Primitive<2>::DIRECTION_TRAN> {

    HorizontalStack(double baseH = 0.0): StackContainerBaseImpl<2, Primitive<2>::DIRECTION_TRAN>(baseH) {}

    /**
     * Check if all children have the same heights.
     * @return @c true only if all children have the same heights
     */
    bool allChildrenHaveSameHeights() const;

    /**
     * Check if all children have the same heights and throw exception it's not true.
     * @throw Exception if not all children have the same heights
     */
    void ensureAllChildrenHaveSameHeights() const {
        if (!allChildrenHaveSameHeights()) throw Exception("Not all children in horizontal stack have the same height");
    }

    /**
     * Add children to stack top.
     * @param el element to add
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(const shared_ptr<ChildType> &el) {
        this->ensureCanHasAsChild(*el);
        return addUnsafe(el);
    }

    /**
     * Add child to stack top.
     * @param el element to add
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint push_back(const shared_ptr<ChildType> &el) { return add(el); }

    /**
     * Add children to stack top.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el element to add
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint addUnsafe(const shared_ptr<ChildType>& el);

    /**
     * Insert children to stack at given position.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el element to insert
     * @param pos position where (before which) child should be inserted
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint insertUnsafe(const shared_ptr<ChildType>& el, const std::size_t pos);

    /**
     * Insert children to stack at given position.
     * @param el element to insert
     * @param pos position where (before which) child should be inserted
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint insert(const shared_ptr<ChildType>& el, const std::size_t pos) {
        this->ensureCanHasAsChild(*el);
        return insertUnsafe(el, pos);
    }

    /**
     * Add children to stack begin, move all other children right.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el element to add
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint push_front_Unsafe(const shared_ptr<ChildType>& el) {
        return insertUnsafe(el, 0);
    }

    /**
     * Add children to stack begin, move all other children right.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el element to add
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint push_front(const shared_ptr<ChildType>& el) {
        this->ensureCanHasAsChild(*el);
        return push_front_Unsafe(el);
    }
};


/**
 * Container which have children in stack/layers.
 */
//TODO copy constructor
template <int dim>
struct StackContainer: public StackContainerBaseImpl<dim> {

    typedef typename chooseType<dim-2, align::Aligner2d<align::DIRECTION_TRAN>, align::Aligner3d<align::DIRECTION_LON, align::DIRECTION_TRAN> >::type Aligner;
    typedef typename chooseType<dim-2, align::Left, align::BackLeft>::type DefaultAligner;

    typedef typename StackContainerBaseImpl<dim>::ChildType ChildType;
    typedef typename StackContainerBaseImpl<dim>::TranslationT TranslationT;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename StackContainerBaseImpl<dim>::Box Box;

    using StackContainerBaseImpl<dim>::shared_from_this;
    using StackContainerBaseImpl<dim>::children;
    using StackContainerBaseImpl<dim>::stackHeights;

  private:
    std::vector< std::unique_ptr<Aligner> > aligners;

    /**
     * Get translation element over given element @p el.
     * @param el element to wrap
     * @param aligner aligner for element
     * @param up_trans translation in growing direction
     * @param elBB bouding box of @p el
     * @return translation over @p el
     */
    shared_ptr<TranslationT> newTranslation(const shared_ptr<ChildType>& el, const Aligner& aligner, double up_trans, const Box& elBB) const {
        shared_ptr<TranslationT> result(new TranslationT(el, Primitive<dim>::ZERO_VEC));
        result->translation.up = up_trans;
        aligner.align(*result, elBB);
        el->fireChanged();
        return result;
    }

    /**
     * Get translation element over given element @p el.
     * @param el element to wrap
     * @param aligner aligner for element
     * @param up_trans translation in growing direction
     * @return translation over @p el
     */
    shared_ptr<TranslationT> newTranslation(const shared_ptr<ChildType>& el, const Aligner& aligner, double up_trans) const {
        shared_ptr<TranslationT> result(new TranslationT(el, Primitive<dim>::ZERO_VEC));
        result->translation.up = up_trans;
        aligner.align(*result);
        return result;
    }

  public:

    /**
     * @param baseHeight height where the first element should start
     */
    explicit StackContainer(const double baseHeight = 0.0): StackContainerBaseImpl<dim>(baseHeight) {}

    //virtual ~StackContainer() { for (auto a: aligners) delete a; }

    /**
     * Insert children to stack at given position.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el element to insert
     * @param pos position where (before which) child should be inserted
     * @param aligner aligner which will be used to calculate horizontal translation of inserted element
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint insertUnsafe(const shared_ptr<ChildType>& el, const std::size_t pos, const Aligner& aligner = DefaultAligner());

    /**
     * Insert children to stack at given position.
     * @param el element to insert
     * @param pos position where (before which) child should be inserted
     * @param aligner aligner which will be used to calculate horizontal translation of inserted element
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint insert(const shared_ptr<ChildType>& el, const std::size_t pos, const Aligner& aligner = DefaultAligner()) {
        this->ensureCanHasAsChild(*el);
        return insertUnsafe(el, pos, aligner);
    }

    /**
     * Add children to stack top.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el element to add
     * @param aligner aligner which will be used to calculate horizontal translation of inserted element
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint addUnsafe(const shared_ptr<ChildType>& el, const Aligner& aligner = DefaultAligner()) {
        double el_translation, next_height;
        auto elBB = el->getBoundingBox();
        this->calcHeight(elBB, stackHeights.back(), el_translation, next_height);
        shared_ptr<TranslationT> trans_geom = newTranslation(el, aligner, el_translation, elBB);
        this->connectOnChildChanged(*trans_geom);
        children.push_back(trans_geom);
        stackHeights.push_back(next_height);
        aligners.push_back(aligner.cloneUnique());
        this->fireChildrenChanged();
        return PathHints::Hint(shared_from_this(), trans_geom);
    }

    /**
     * Add children to stack top.
     * @param el element to add
     * @param aligner aligner which will be used to calculate horizontal translation of inserted element
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint add(const shared_ptr<ChildType> &el, const Aligner& aligner = DefaultAligner()) {
        this->ensureCanHasAsChild(*el);
        return addUnsafe(el, aligner);
    }

    /**
     * Add child to stack top.
     * @param el element to add
     * @param aligner aligner for horizontal translation of element
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint push_back(const shared_ptr<ChildType> &el, const Aligner& aligner = DefaultAligner()) { return add(el, aligner); }

    /**
     * Add children to stack bottom, move all other children up.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after adding the new child.
     * @param el element to add
     * @param tran_translation horizontal translation of element
     * @return path hint, see @ref geometry_paths
     */
    PathHints::Hint push_front_Unsafe(const shared_ptr<ChildType>& el, const Aligner& aligner = DefaultAligner()) {
        return insertUnsafe(el, 0, aligner);
    }

    /**
     * Add child to stack bottom, move all other children up.
     * @param el element to add
     * @param tran_translation horizontal translation of element
     * @return path hint, see @ref geometry_paths
     * @throw CyclicReferenceException if adding the new child cause inception of cycle in geometry graph
     */
    PathHints::Hint push_front(const shared_ptr<ChildType>& el, const Aligner& aligner = DefaultAligner()) {
        this->ensureCanHasAsChild(*el);
        return push_front_Unsafe(el, aligner);
    }

    const Aligner& getAlignerAt(std::size_t child_nr) const {
        this->ensureIsValidChildNr(child_nr, "getAlignerAt");
        return *aligners[child_nr];
    }

    void setAlignerAt(std::size_t child_nr, const Aligner& aligner);

    /*
     * Set new aligner.
     * @param child_nr (real) child number for which aligner will be set
     * @param aligner new aligner for given child, this pointer will be delete by this stack and it can be used only in one stack, for one child
     */
    /*void setAlignerAtMove(std::size_t child_nr, Aligner* aligner) {
        this->ensureIsValidChildNr(child_nr, "setAlignerAtMove");
        if (aligners[child_nr] == aligner) return; //protected for self assign
        aligners[child_nr] = aligner;
        aligners[child_nr]->align(*children[child_nr]);
        this->fireChanged(GeometryElement::Event::RESIZE);
    }*/

    virtual bool removeIfTUnsafe(const std::function<bool(const shared_ptr<TranslationT>& c)>& predicate);

    virtual void removeAtUnsafe(std::size_t index);

};

template <int dim>
class MultiStackContainer: public StackContainer<dim> {

    ///Type of parent class of this.
    typedef StackContainer<dim> UpperClass;

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

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename UpperClass::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename UpperClass::Box Box;

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
    const bool reduceHeight(double& height) const;

  public:

    /// How many times all stack is repeated.
    unsigned repeat_count;

    /**
     * @param repeat_count how many times stack should be repeated, must be 1 or more
     * @param baseHeight height where the first element should start
     */
    explicit MultiStackContainer(unsigned repeat_count = 1, const double baseHeight = 0.0): UpperClass(baseHeight), repeat_count(repeat_count) {}

    //this is not used but, just for case redefine UpperClass::getChildForHeight
    /*const shared_ptr<TranslationT> getChildForHeight(double height) const {
        if (!reduceHeight(height)) return nullptr;
        return UpperClass::getChildForHeight(height);
    }*/

    virtual bool intersect(const Box& area) const;

    virtual Box getBoundingBox() const {
        Box result = UpperClass::getBoundingBox();
        result.upper.up += result.sizeUp() * (repeat_count-1);
        return result;
    }

    virtual void getBoundingBoxesToVec(const GeometryElement::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const;

    virtual void getElementsToVec(const GeometryElement::Predicate& predicate, std::vector< shared_ptr<const GeometryElement> >& dest, const PathHints* path = 0) const;

    virtual void getPositionsToVec(const GeometryElement::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path = 0) const;

    /*virtual std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > getLeafsWithTranslations() const {
        std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > result = UpperClass::getLeafsWithTranslations();
        std::size_t size = result.size();   //oryginal size
        const double stackHeight = stackHeights.back() - stackHeights.front();
        for (unsigned r = 1; r < repeat_count; ++r) {
            for (std::size_t i = 0; i < size; ++i) {
                result.push_back(result[i]);
                std::get<1>(result.back()).up += stackHeight * r;
            }
        }
        return result;
    }*/

    virtual GeometryElement::Subtree findPathsTo(const GeometryElement& el, const PathHints* path = 0) const;

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

    virtual std::size_t getChildrenCount() const { return children.size() * repeat_count; }

    virtual shared_ptr<GeometryElement> getChildAt(std::size_t child_nr) const;

    virtual std::size_t getRealChildrenCount() const {
        return StackContainer<dim>::getChildrenCount();
    }

    virtual shared_ptr<GeometryElement> getRealChildAt(std::size_t child_nr) const {
        return StackContainer<dim>::getChildAt(child_nr);
    }

    void setRepeatCount(unsigned new_repeat_count) {
        if (repeat_count == new_repeat_count) return;
        repeat_count = new_repeat_count;
        this->fireChildrenChanged();
    }

};


}   // namespace plask


#endif // STACK_H
