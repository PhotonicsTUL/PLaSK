#ifndef PLASK__GEOMETRY_ELEMENT_H
#define PLASK__GEOMETRY_ELEMENT_H

#include <memory>

#include "../material/material.h"
#include "primitives.h"

namespace plask {

enum GeometryElementType {
    GE_TYPE_LEAF = 0,         ///<leaf element (has no child)
    GE_TYPE_TRANSFORM = 1,    ///<transform element (has one child)
    GE_TYPE_CONTAINER = 2     ///<container (more than one child)
};

/**
 * Base class for all geometries.
 */
struct GeometryElement {
    
    /**
     * Check if geometry is: leaf, transform or container type element.
     * @return type of this element
     */
    virtual GeometryElementType getType() const;
    
    /**
     * Virtual destructor. Do nothing.
     */
    virtual ~GeometryElement() {}
    
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
    
};

/**
 * Template for base class for all leaf nodes.
 */
template < int dim >
struct GeometryElementLeaf: public GeometryElementD<dim> {
    
    typedef typename GeometryElementD<dim>::Vec Vec;
    
    std::shared_ptr<Material> material;
    
    GeometryElementLeaf<dim>(std::shared_ptr<Material> material): material(material) {}
    
    virtual GeometryElementType getType() const { return GE_TYPE_LEAF; }
    
    virtual std::shared_ptr<Material> getMaterial(const Vec& p) const {
        return includes(p) ? material : nullptr;
    }
    
};

/**
 * Template for base class for all transform nodes.
 */
template < int dim, typename ChildType = GeometryElementD<dim> >
struct GeometryElementTransform: public GeometryElementD<dim> {
    
    GeometryElementTransform(ChildType* child): _child(child) {}
    
    virtual GeometryElementType getType() const { return GE_TYPE_TRANSFORM; }
    
    ChildType& child() { return *child; }   //TODO check if child != nullptr
    
    protected:
    ChildType* _child;
    
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
