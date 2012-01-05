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
    typedef std::map<GeometryElement*, GeometryElement*> HintMap;
    //TODO co z wieloma ojcami, może powinno być GeometryElement* -> set<GeometryElement*> (bo kontenery też mogą mieć wiele ojców) albo mapy w obie strony
    //TODO może GeometryElement* -> int

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
    void addHint(GeometryElement* container, GeometryElement* child);

    /**
     * Get child for given container.
     * @return child for given container or @c nullptr if there is no hint for given container
     */
    GeometryElement* getChild(GeometryElement* container) const;

};

/**
 * Template which implement container using stl container of pointers to some geometry elements objects (Translation by default).
 * @tparam dim GeometryElementContainer dimension
 * @tparam container_type container of pointers to children
 */
template <int dim, typename container_type = std::vector<Translation<dim>*> >
struct GeometryElementContainerImpl: public GeometryElementContainer<dim> {

    typedef typename GeometryElementContainer<dim>::DVec DVec;
    typedef typename GeometryElementContainer<dim>::Rect Rect;

protected:
    container_type children;

public:
    ~GeometryElementContainerImpl() {
        for (auto child: children) delete child;
    }

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
     * Check children in reverse order and check if any returns material.
     * @return material or @c nullptr
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

};

/**
 * Geometry elements container in which all child is connected with translation vector.
 */
//TODO some implementation are naive, and can be done faster with some caches
template < int dim >
struct TranslationContainer: public GeometryElementContainerImpl<dim> {

    typedef typename GeometryElementContainer<dim>::DVec DVec;
    typedef typename GeometryElementContainer<dim>::Rect Rect;
    typedef GeometryElementD<dim> ChildType;
    typedef Translation<dim> TranslationT;

    using GeometryElementContainerImpl<dim>::children;

    PathHints::Hint add(ChildType* el, const DVec& translation = Primitive<dim>::ZERO_VEC) {
        TranslationT* trans_geom = new TranslationT(el, translation);
        children.push_back(trans_geom);
        return PathHints::Hint(this, trans_geom);
    }

};

/**
 * Common code for stack containers (which have children in stack/layers).
 */
template <int dim>
struct StackContainerBaseImpl: public GeometryElementContainerImpl<dim> {

    typedef typename GeometryElementContainer<dim>::DVec DVec;
    typedef typename GeometryElementContainer<dim>::Rect Rect;
    typedef GeometryElementD<dim> ChildType;
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
    virtual const TranslationT* getChildForHeight(double height) const {
        auto it = std::lower_bound(stackHeights.begin(), stackHeights.end(), height);
        if (it == stackHeights.end() || it == stackHeights.begin()) return nullptr;
        return children[it-stackHeights.begin()-1];
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

};


/**
 * 2d container which have children in stack/layers.
 */
struct StackContainer2d: public StackContainerBaseImpl<2> {

    using StackContainerBaseImpl<2>::children;

    /**
     * @param baseHeight height where should start first element
     */
    explicit StackContainer2d(const double baseHeight = 0.0);

    /**
     * Add children to stack top.
     * @param el element to add
     * @param tran_translation horizontal translation of element
     * @return path hint
     */
    PathHints::Hint push_back(ChildType* el, const double tran_translation = 0.0);

    /**
     * Add children to stack top.
     * @param el element to add
     * @param tran_translation horizontal translation of element
     * @return path hint
     */
    PathHints::Hint add(ChildType* el, const double tran_translation = 0.0) { return push_back(el, tran_translation); }

};

/**
 * 3d container which have children in stack/layers.
 */
struct StackContainer3d: public StackContainerBaseImpl<3> {

    using StackContainerBaseImpl<3>::children;

    /**
     * @param baseHeight height where should start first element
     */
    explicit StackContainer3d(const double baseHeight = 0.0);

    /**
     * Add children to stack top.
     * @param el element to add
     * @param lon_translation, tran_translation horizontal translation of element
     * @return path hint
     */
    PathHints::Hint push_back(ChildType* el, const double lon_translation = 0.0, const double tran_translation = 0.0);

    /**
     * Add children to stack top.
     * @param el element to add
     * @param lon_translation, tran_translation horizontal translation of element
     * @return path hint
     */
    PathHints::Hint add(ChildType* el, const double lon_translation = 0.0, const double tran_translation = 0.0) { return push_back(el, lon_translation, tran_translation); }

};

template <int dim>
class MultiStackContainer: public chooseType<dim-2, StackContainer2d, StackContainer3d>::type {
    
    typedef typename chooseType<dim-2, StackContainer2d, StackContainer3d>::type UpperClass;
    
    static double modulo(double a, double divider) { 
        return a - static_cast<double>( static_cast<int>( a / divider ) ) * divider;
    }
    
public:
    using UpperClass::getChildForHeight;
    using UpperClass::stackHeights;
    using UpperClass::children;
    
    typedef typename MultiStackContainer::Rect Rect;
    
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
    
public:
    
    ///How muny times all stack is repeat.
    unsigned repeat_count;
    
    MultiStackContainer(const double baseHeight = 0.0, unsigned repeat_count = 1): UpperClass(baseHeight), repeat_count(repeat_count) {}
    
    //redefinision of virtual class makes many geometry elment methods impl. fine
    const typename UpperClass::TranslationT* getChildForHeight(double height) const {
        const double zeroBasedStackHeight = stackHeights.back() - stackHeights.front();
        const double zeroBasedRequestHeight = height - stackHeights.front();
        if (zeroBasedRequestHeight < 0.0 || zeroBasedRequestHeight > zeroBasedStackHeight * repeat_count)
            return nullptr;
            //throw OutOfBoundException("MultiStack::getChildForHeight", "height", height, baseHeight, zeroBasedStackHeight * repeat_count + baseHeight);
        return UpperClass::getChildForHeight(modulo(zeroBasedRequestHeight, zeroBasedStackHeight) + stackHeights.front());
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
    
};



}	// namespace plask

#endif // PLASK__GEOMETRY_CONTAINER_H
