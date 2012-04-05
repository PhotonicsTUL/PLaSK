#ifndef PLASK__GEOMETRY_TRANSFORM_H
#define PLASK__GEOMETRY_TRANSFORM_H

#include "element.h"
#include <boost/bind.hpp>
//#include <functional>

namespace plask {

/**
 * Template of base class for all transform nodes.
 * Transform node has exactly one child node and represent element which is equal to child after transform.
 * @tparam dim number of dimensions of this element
 * @tparam Child_Type type of child, can be in space with different number of dimensions than this is (in such case see @ref GeometryElementTransformSpace).
 */
template < int dim, typename Child_Type = GeometryElementD<dim> >
struct GeometryElementTransform: public GeometryElementD<dim> {

    typedef Child_Type ChildType;

    explicit GeometryElementTransform(shared_ptr<ChildType> child = nullptr): _child(child) { connectOnChildChanged(); }

    explicit GeometryElementTransform(ChildType& child): _child(static_pointer_cast<ChildType>(child.shared_from_this())) { connectOnChildChanged(); }

    virtual ~GeometryElementTransform() { disconnectOnChildChanged(); }

    virtual GeometryElement::Type getType() const { return GeometryElement::TYPE_TRANSFORM; }

    virtual void getLeafsToVec(std::vector< shared_ptr<const GeometryElement> >& dest) const {
        getChild()->getLeafsToVec(dest);
    }

    /// Called by child.change signal, call this change
    virtual void onChildChanged(const GeometryElement::Event& evt) {
        this->fireChanged(evt.flagsForParent());
    }

    /// Connect onChildChanged to current child change signal
    void connectOnChildChanged() {
        if (_child)
            _child->changed.connect(
                boost::bind(&GeometryElementTransform<dim, Child_Type>::onChildChanged, this, _1)
            );
    }

    /// Disconnect onChildChanged from current child change signal
    void disconnectOnChildChanged() {
        if (_child)
            _child->changed.disconnect(
                boost::bind(&GeometryElementTransform<dim, Child_Type>::onChildChanged, this, _1)
            );
    }

    /**
     * Get child.
     * @return child
     */
    inline shared_ptr<ChildType> getChild() const { return _child; }

    /**
     * Set new child.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after setting the new child.
     * @param child new child
     */
    void setChildUnsafe(const shared_ptr<ChildType>& child) {
        if (child == _child) return;
        disconnectOnChildChanged();
        _child = child;
        connectOnChildChanged();
    }

    /**
     * Set new child.
     * @param child new child
     * @throw CyclicReferenceException if set new child cause inception of cycle in geometry graph
     */
    void setChild(const shared_ptr<ChildType>& child) {
        this->ensureCanHasAsChild(*child);
        setChildUnsafe(child);
    }

    /**
     * @return @c true only if child is set (not null)
     */
    bool hasChild() const { return _child != nullptr; }

    /**
     * Throw NoChildException if child is not set.
     */
    virtual void validate() const {
        if (!hasChild()) throw NoChildException();
    }

    virtual bool isInSubtree(const GeometryElement& el) const {
        return &el == this || (hasChild() && _child->isInSubtree(el));
    }

    virtual GeometryElement::Subtree findPathsTo(const GeometryElement& el, const PathHints* path = 0) const {
        if (this == &el) return GeometryElement::Subtree(this->shared_from_this());
        if (!_child) GeometryElement::Subtree();
        GeometryElement::Subtree e = _child->findPathsTo(el, path);
        if (e.empty()) return GeometryElement::Subtree();
        GeometryElement::Subtree result(this->shared_from_this());
        result.children.push_back(std::move(e));
        return result;
    }

    virtual std::size_t getChildrenCount() const { return hasChild() ? 1 : 0; }

    virtual shared_ptr<GeometryElement> getChildAt(std::size_t child_nr) const {
        if (!hasChild() || child_nr > 0) throw OutOfBoundException("GeometryElementTransform::getChildAt", "child_nr");
        return _child;
    }

    /**
     * Get shallow copy of this.
     * @return shallow copy of this
     */
    virtual shared_ptr<GeometryElementTransform<dim, Child_Type>> shallowCopy() const = 0;

    shared_ptr<GeometryElementTransform<dim, Child_Type>> shallowCopy(const shared_ptr<ChildType>& child) const {
        shared_ptr<GeometryElementTransform<dim, Child_Type>> result = shallowCopy();
        result->setChild(child);
        return result;
    }

    virtual shared_ptr<const GeometryElement> changedVersion(const GeometryElement::Changer& changer, Vec<3, double>* translation = 0) const {
        shared_ptr<const GeometryElement> result(this->shared_from_this());
        if (changer.apply(result, translation) || !hasChild()) return result;
        shared_ptr<const GeometryElement> new_child = _child->changedVersion(changer, translation);
        return new_child == _child ? result : shallowCopy(const_pointer_cast<ChildType>(dynamic_pointer_cast<const ChildType>(new_child)));
    }

    protected:
    shared_ptr<ChildType> _child;

};

/**
 * Template of base class for all transformations which change the space between its parent and child.
 * @tparam this_dim number of dimensions of this element
 * @tparam child_dim number of dimensions of child element
 * @tparam ChildType type of child, should be in space with @a child_dim number of dimensions
 */
template < int this_dim, int child_dim = 5-this_dim, typename ChildType = GeometryElementD<child_dim> >
struct GeometryElementTransformSpace: public GeometryElementTransform<this_dim, ChildType> {

    typedef typename ChildType::Box ChildBox;
    typedef typename ChildType::DVec ChildVec;
    typedef typename GeometryElementTransform<this_dim, ChildType>::DVec DVec;
    using GeometryElementTransform<this_dim, ChildType>::getChild;

    explicit GeometryElementTransformSpace(shared_ptr<ChildType> child = shared_ptr<ChildType>()): GeometryElementTransform<this_dim, ChildType>(child) {}

    /// @return GE_TYPE_SPACE_CHANGER
    virtual GeometryElement::Type getType() const { return GeometryElement::TYPE_SPACE_CHANGER; }

    virtual std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > getLeafsWithTranslations() const {
        std::vector< shared_ptr<const GeometryElement> > v = getChild()->getLeafs();
        std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > result(v.size());
        std::transform(v.begin(), v.end(), result.begin(), [](shared_ptr<const GeometryElement> e) {
            return std::make_pair(e, Primitive<this_dim>::NAN_VEC);
        });
        return result;
    }

};

/**
 * Represent geometry element equal to its child translated by vector.
 */
template <int dim>
struct Translation: public GeometryElementTransform<dim> {

    typedef typename GeometryElementTransform<dim>::ChildType ChildType;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryElementTransform<dim>::DVec DVec;

    /// Box type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryElementTransform<dim>::Box Box;

    using GeometryElementTransform<dim>::getChild;

    /**
     * Translation vector.
     */
    DVec translation;

    //Translation(const Translation<dim>& translation) = default;

    /**
     * @param child child geometry element, element to translate
     * @param translation translation
     */
    explicit Translation(shared_ptr< GeometryElementD<dim> > child = shared_ptr< GeometryElementD<dim> >(), const DVec& translation = Primitive<dim>::ZERO_VEC)
        : GeometryElementTransform<dim>(child), translation(translation) {}

    explicit Translation(GeometryElementD<dim>& child, const DVec& translation = Primitive<dim>::ZERO_VEC)
        : GeometryElementTransform<dim>(child), translation(translation) {}

    virtual Box getBoundingBox() const {
        return getChild()->getBoundingBox().translated(translation);
    }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        return getChild()->getMaterial(p-translation);
    }

    virtual bool inside(const DVec& p) const {
        return getChild()->inside(p-translation);
    }

    virtual bool intersect(const Box& area) const {
        return getChild()->intersect(area.translated(-translation));
    }

    /*virtual void getLeafsInfoToVec(std::vector< std::tuple<shared_ptr<const GeometryElement>, Box, DVec> >& dest, const PathHints* path = 0) const {
        const std::size_t old_size = dest.size();
        getChild()->getLeafsInfoToVec(dest, path);
        for (auto i = dest.begin() + old_size; i != dest.end(); ++i) {
            std::get<1>(*i).translate(translation);
            std::get<2>(*i) += translation;
        }
    }*/

    virtual void getBoundingBoxesToVec(const GeometryElement::Predicate& predicate, std::vector<Box>& dest, const PathHints* path = 0) const {
        if (predicate(*this)) {
            dest.push_back(getBoundingBox());
            return;
        }
        std::vector<Box> result = getChild()->getBoundingBoxes(predicate, path);
        dest.reserve(dest.size() + result.size());
        for (Box& r: result) dest.push_back(r.translated(translation));
    }

    virtual std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > getLeafsWithTranslations() const {
        std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > result = getChild()->getLeafsWithTranslations();
        for (std::tuple<shared_ptr<const GeometryElement>, DVec>& r: result) std::get<1>(r) += translation;
        return result;
    }

    /**
     * Get shallow copy of this.
     * @return shallow copy of this
     */
    shared_ptr<Translation<dim>> copyShallow() const {
         return shared_ptr<Translation<dim>>(new Translation<dim>(getChild(), translation));
    }

    virtual shared_ptr<GeometryElementTransform<dim>> shallowCopy() const {
        return copyShallow();
    }

    virtual shared_ptr<const GeometryElement> changedVersion(const GeometryElement::Changer& changer, Vec<3, double>* translation = 0) const {
        shared_ptr<const GeometryElement> result(this->shared_from_this());
        if (changer.apply(result, translation) || !this->hasChild()) return result;
        Vec<3, double> returned_translation(0.0, 0.0, 0.0);
        shared_ptr<const GeometryElement> new_child = this->getChild()->changedVersion(changer, &returned_translation);
        Vec<dim, double> translation_we_will_do = vec<dim, double>(returned_translation);
        if (new_child == getChild() && translation_we_will_do == Primitive<dim>::ZERO_VEC) return result;
        if (translation)    //we will change translation (partially if dim==2) internaly, so we recommend no extra translation
            *translation = returned_translation - vec<3, double>(translation_we_will_do); //still we can recommend translation in third direction
        return shared_ptr<GeometryElement>(
            new Translation<dim>(const_pointer_cast<ChildType>(dynamic_pointer_cast<const ChildType>(new_child)),
            this->translation + translation_we_will_do) );
    }

    /**
     * Get shallow, moved copy of this.
     * @param new_translation translation vector of copy
     */
    shared_ptr<Translation<dim>> copyShallow(const DVec& new_translation) const {
        return shared_ptr<Translation<dim>>(new Translation<dim>(getChild(), new_translation));
    }

};

}       // namespace plask

#endif // PLASK__GEOMETRY_TRANSFORM_H
