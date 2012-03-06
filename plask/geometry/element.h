#ifndef PLASK__GEOMETRY_ELEMENT_H
#define PLASK__GEOMETRY_ELEMENT_H

/** @file
This file includes base classes for geometries elements.
*/


#include <vector>
#include <tuple>

#include "../material/material.h"
#include "primitives.h"

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
//TODO child number, child by index (?), child iterator
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
     * Subclasses of this can includes additional informations about specific type of event.
     */
    struct Event {
        
        enum Type { SHAPE = 1, MATERIAL = 1<<1, DELETE = 1<<2 };    //??
        
    private:
        GeometryElement& _source;
        Type _type;
        
    public:

        const GeometryElement& source() const { return _source; }
        const Type type() const { return _type; }
        
        Event(GeometryElement& source, Type type): _source(source), _type(type) {}
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
        bool isWithBranches() const;

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
         * @param to_change[in,out] pointer to element which eventualy will be changed (in such case pointer after call can point to other geometry element)
         * @param translation[out] optional, extra translation for element after change (in case of 2d object caller read only tran and up components of vector)
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
    
    boost::signals2::signal<void(const Event&)> changed;

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
     * Get all leafs in subtree with this in root.
     * @return all leafs in subtree with this in root
     */
    std::vector< shared_ptr<const GeometryElement> > getLeafs() const {
        std::vector< shared_ptr<const GeometryElement> > result;
        getLeafsToVec(result);
        return result;
    }

    virtual std::size_t getChildCount() const = 0;

    virtual shared_ptr<GeometryElement> getChildAt(std::size_t child_nr) const = 0;

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
    typedef typename Primitive<dim>::Rect Rect;
    typedef typename Primitive<dim>::DVec DVec;

    int getDimensionsCount() const { return dimensions; }

    //virtual Rect getBoundingBox() const;

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
    virtual bool intersect(const Rect& area) const = 0;

    /**
     * Calculate minimal rectangle which includes all points of geometry element.
     * @return calculated rectangle
     */
    virtual Rect getBoundingBox() const = 0;

    virtual DVec getBoundingBoxSize() const { return getBoundingBox().size(); }

    /**
     * Return material in a given point inside the geometry element
     * @param p point
     * @return material in given point, or @c nullptr if this GeometryElement not includes point @a p
     */
    virtual shared_ptr<Material> getMaterial(const DVec& p) const = 0;

    //virtual std::vector<Material*> getMaterials(Mesh);        ??

    //virtual void getLeafsInfoToVec(std::vector<std::tuple<shared_ptr<const GeometryElement>, Rect, DVec>>& dest, const PathHints* path = 0) const = 0;

    /**
     * Calculate and append to vector bounding boxes of all leafs, optionaly showed by path.
     * @param dest place to add result
     * @param path path fragments, optional
     */
    virtual void getLeafsBoundingBoxesToVec(std::vector<Rect>& dest, const PathHints* path = 0) const = 0;

    /**
     * Calculate bounding boxes of all leafs, optionaly showed by path.
     * @param path path fragments, optional
     * @return bounding boxes of all leafs
     */
    std::vector<Rect> getLeafsBoundingBoxes(const PathHints* path = 0) const {
        std::vector<Rect> result;
        getLeafsBoundingBoxesToVec(result, path);
        return result;
    }

    /**
     * Calculate bounding boxes of all leafs, optionaly showed by path.
     * @param path path fragments
     * @return bounding boxes of all leafs
     */
    std::vector<Rect> getLeafsBoundingBoxes(const PathHints& path) const {
        std::vector<Rect> result;
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

/**
 * Template of base classes for all leaf nodes.
 * @tparam dim number of dimensions
 */
template < int dim >
struct GeometryElementLeaf: public GeometryElementD<dim> {

    typedef typename GeometryElementD<dim>::DVec DVec;
    typedef typename GeometryElementD<dim>::Rect Rect;
    using GeometryElementD<dim>::getBoundingBox;
    using GeometryElementD<dim>::shared_from_this;

    shared_ptr<Material> material;

    GeometryElementLeaf<dim>(shared_ptr<Material> material): material(material) {}

    virtual GeometryElement::Type getType() const { return GeometryElement::TYPE_LEAF; }

    virtual shared_ptr<Material> getMaterial(const DVec& p) const {
        return this->inside(p) ? material : shared_ptr<Material>();
    }

    virtual void getLeafsInfoToVec(std::vector<std::tuple<shared_ptr<const GeometryElement>, Rect, DVec>>& dest, const PathHints* path = 0) const {
        dest.push_back( std::tuple<shared_ptr<const GeometryElement>, Rect, DVec>(this->shared_from_this(), this->getBoundingBox(), Primitive<dim>::ZERO_VEC) );
    }

    virtual void getLeafsBoundingBoxesToVec(std::vector<Rect>& dest, const PathHints* path = 0) const {
        dest.push_back(this->getBoundingBox());
    }

    inline std::vector<Rect> getLeafsBoundingBoxes() const {
        return { this->getBoundingBox() };
    }

    inline std::vector<Rect> getLeafsBoundingBoxes(const PathHints&) const {
        return { this->getBoundingBox() };
    }

    virtual void getLeafsToVec(std::vector< shared_ptr<const GeometryElement> >& dest) const {
        dest.push_back(this->shared_from_this());
    }
    
    inline std::vector< shared_ptr<const GeometryElement> > getLeafs() const {
        return { this->shared_from_this() };
    }
    
    virtual std::vector< std::tuple<shared_ptr<const GeometryElement>, DVec> > getLeafsWithTranslations() const {
        return { std::make_pair(shared_from_this(), Primitive<dim>::ZERO_VEC) };
    }

    virtual bool isInSubtree(const GeometryElement& el) const {
        return &el == this;
    }

    virtual GeometryElement::Subtree findPathsTo(const GeometryElement& el, const PathHints* path = 0) const {
        return GeometryElement::Subtree( &el == this ? this->shared_from_this() : shared_ptr<const GeometryElement>() );
    }

    virtual std::size_t getChildCount() const { return 0; }

    virtual shared_ptr<GeometryElement> getChildAt(std::size_t child_nr) const {
        throw OutOfBoundException("GeometryElementLeaf::getChildAt", "child_nr");
    }

    virtual shared_ptr<const GeometryElement> changedVersion(const GeometryElement::Changer& changer, Vec<3, double>* translation = 0) const {
        shared_ptr<const GeometryElement> result(this->shared_from_this());
        changer.apply(result, translation);
        return result;
    }

};

/**
 * Template of base class for all transform nodes.
 * Transform node has exactly one child node and represent element which is equal to child after transform.
 * @tparam dim number of dimensions of this element
 * @tparam Child_Type type of child, can be in space with different number of dimensions than this is (in such case see @ref GeometryElementChangeSpace).
 */
template < int dim, typename Child_Type = GeometryElementD<dim> >
struct GeometryElementTransform: public GeometryElementD<dim> {

    typedef Child_Type ChildType;

    explicit GeometryElementTransform(shared_ptr<ChildType> child = nullptr): _child(child) {}

    explicit GeometryElementTransform(ChildType& child): _child(static_pointer_cast<ChildType>(child.shared_from_this())) {}

    virtual GeometryElement::Type getType() const { return GeometryElement::TYPE_TRANSFORM; }

    virtual void getLeafsToVec(std::vector< shared_ptr<const GeometryElement> >& dest) const {
        getChild()->getLeafsToVec(dest);
    }

    /**
     * Get child.
     * @return child
     */
    shared_ptr<ChildType> getChild() const { return _child; }

    /**
     * Set new child.
     * This method is fast but also unsafe because it doesn't ensure that there will be no cycle in geometry graph after setting the new child.
     * @param child new child
     */
    void setChildUnsafe(const shared_ptr<ChildType>& child) { _child = child; }

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

    virtual std::size_t getChildCount() const { return hasChild() ? 1 : 0; }

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
 * Template of base class for all space changer nodes.
 * Space changer if transform node which is in space with different number of dimensions than its child.
 * @tparam this_dim number of dimensions of this element
 * @tparam child_dim number of dimensions of child element
 * @tparam ChildType type of child, should be in space with @a child_dim number of dimensions
 */
template < int this_dim, int child_dim = 5 - this_dim, typename ChildType = GeometryElementD<child_dim> >
struct GeometryElementChangeSpace: public GeometryElementTransform<this_dim, ChildType> {

    typedef typename ChildType::Rect ChildRect;
    typedef typename ChildType::DVec ChildVec;
    typedef typename GeometryElementTransform<this_dim, ChildType>::DVec DVec;
    using GeometryElementTransform<this_dim, ChildType>::getChild;

    explicit GeometryElementChangeSpace(shared_ptr<ChildType> child = shared_ptr<ChildType>()): GeometryElementTransform<this_dim, ChildType>(child) {}

    ///@return GE_TYPE_SPACE_CHANGER
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
 * Template of base class for all container nodes.
 * Container nodes can include one or more child nodes.
 */
template < int dim >
struct GeometryElementContainer: public GeometryElementD<dim> {

    ///@return GE_TYPE_CONTAINER
    virtual GeometryElement::Type getType() const { return GeometryElement::TYPE_CONTAINER; }

};

}       // namespace plask

#endif	// PLASK__GEOMETRY_ELEMENT_H
