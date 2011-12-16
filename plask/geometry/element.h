#ifndef PLASK__GEOMETRY_ELEMENT_H
#define PLASK__GEOMETRY_ELEMENT_H

/** @file
This file includes base classes for geometries elements.
*/


#include <memory>
#include <vector>

#include "../material/material.h"
#include "primitives.h"

namespace plask {

enum GeometryElementType {
    GE_TYPE_LEAF = 0,         ///<leaf element (has no child)
    GE_TYPE_TRANSFORM = 1,    ///<transform element (has one child)
    GE_TYPE_CONTAINER = 2,    ///<container (more than one child)
    GE_TYPE_SPACE_CHANGER = 3 ///<transform element changing its space, typically changing number of dimensions (has one child)
};

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
struct GeometryElement {
    
    /**
     * Check if geometry is: leaf, transform or container type element.
     * @return type of this element
     */
    virtual GeometryElementType getType() const = 0;

    /**
     * Check if element is ready for calculation.
     * Throw exception if element is in bad state and can't be used in calculations, for example has not required children, etc.
     * Default implementation do nothing, but inharited class can change this bechaviour.
     * @throw Exception if element is not ready for calculation
     */
    virtual void validate() const throw (Exception) {}
    
    /**
     * Virtual destructor. Do nothing.
     */
    virtual ~GeometryElement() {}
    
    //virtual GeometryTransform getTransform()
    
};

template < int dimensions >
struct GeometryElementD: public GeometryElement {
    
    static const int dim = dimensions;
    typedef typename Primitive<dim>::Rect Rect;
    typedef typename Primitive<dim>::Vec Vec;
    
    //virtual Rect getBoundingBox() const;
    
    /**
     * Check if geometry includes point.
     * @param p point
     * @return true only if this geometry includes point @a p
     */
    virtual bool inside(const Vec& p) const = 0;
    
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
    
    virtual Vec getBoundingBoxSize() const { return getBoundingBox().size(); }
    
    //virtual GeometryElementD<dim>* getLeaf(const Vec& p) const; //shared_ptr?
    
    //virtual std::vector<GeometryElementD<dim>*> getLeafs() const;     //shared_ptr?
    
    /**
     * @param p point
     * @return material in given point, or @c nullptr if this GeometryElement not includes point @a p
     */
    virtual std::shared_ptr<Material> getMaterial(const Vec& p) const = 0;
    
    //virtual std::vector<Material*> getMaterials(Mesh);        ??
    
    /**
     * Calculate bounding boxes of all leafs.
     * @return bounding boxes of all leafs
     */
    virtual std::vector<Rect> getLeafsBoundingBoxes() const = 0;
    
};

/**
 * Template for base class for all leaf nodes.
 */
template < int dim >
struct GeometryElementLeaf: public GeometryElementD<dim> {
    
    typedef typename GeometryElementD<dim>::Vec Vec;
    typedef typename GeometryElementD<dim>::Rect Rect;
    
    std::shared_ptr<Material> material;
    
    GeometryElementLeaf<dim>(std::shared_ptr<Material> material): material(material) {}
    
    virtual GeometryElementType getType() const { return GE_TYPE_LEAF; }
    
    virtual std::shared_ptr<Material> getMaterial(const Vec& p) const {
        return includes(p) ? material : nullptr;
    }
    
    virtual std::vector<Rect> getLeafsBoundingBoxes() const {
        return { GeometryElementD<dim>::getBoundingBox() };
    }
    
};

/**
 * Template for base class for all transform nodes.
 */
template < int dim, typename ChildType = GeometryElementD<dim> >
struct GeometryElementTransform: public GeometryElementD<dim> {
    
    explicit GeometryElementTransform(ChildType* child = 0): _child(child) {}
    
    virtual GeometryElementType getType() const { return GE_TYPE_TRANSFORM; }
    
    ChildType& getChild() { return *_child; }

    const ChildType& getChild() const { return *_child; }

    void setChild(ChildType* child) { _child = child; }

    void setChild(ChildType& child) { _child = &child; }

    bool hasChild() const { return _child != nullptr; }

    virtual void validate() const throw (Exception) {
        if (!hasChild()) throw NoChildException();
    }
    
    protected:
    ChildType* _child;
    
};

/**
 * Template for base class for all space changer nodes.
 */
template < int this_dim, int child_dim, typename ChildType = GeometryElementD<child_dim> >
struct GeometryElementChangeSpace: public GeometryElementTransform<this_dim, ChildType> {

    typedef typename ChildType::Rect ChildRect;
    typedef typename ChildType::Vec ChildVec;

    explicit GeometryElementChangeSpace(ChildType* child = 0): GeometryElementTransform<this_dim, ChildType>(child) {}

    virtual GeometryElementType getType() const { return GE_TYPE_SPACE_CHANGER; }

};

/**
 * Template for base class for all container nodes.
 */
template < int dim >
struct GeometryElementContainer: public GeometryElementD<dim> {
    
    virtual GeometryElementType getType() const { return GE_TYPE_CONTAINER; }
    
};

}       // namespace plask

#endif	// PLASK__GEOMETRY_ELEMENT_H
