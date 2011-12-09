#ifndef PLASK__GEOMETRY_H
#define PLASK__GEOMETRY_H

#include <memory>

#include "../material/material.h"

namespace plask {

/**
 * Base class for all geometries.
 */
struct GeometryElement {
    
    /**
     * Check if geometry is leaf.
     * Default implementation return @c false, so only leafs should overwrite this.
     * @return @c true only if geometry is leaf
     */
    virtual bool isLeaf() const { return false; }
    
    /**
     * Virtual destructor. Do nothing.
     */
    virtual ~GeometryElement() {}
    
};

template < int dim >
struct GeometryElementD: public GeometryElement {
    
    typedef typename Primitive<dim>::Rect Rect;
    typedef typename Primitive<dim>::Vec Vec;
    static const int dim = dim;
    
    //virtual Rect getBoundingBox() const;
    
    /**
     * Check if geometry includes point.
     * @param p point
     * @return true only if this geometry includes point @a p
     */
    virtual bool includes(const Vec& p) const = 0;
    
    /**
     * Check if geometry includes some point from given @a area.
     * @param area rectangular area
     * @return true only if this geometry includes some points from @a area
     */
    virtual bool includes(const Rect& area) const = 0;
    
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
struct GeometryElementLeaf: GeometryElementD<dim> {
    
    std::shared_ptr<Material> material;
    
    virtual bool isLeaf() const { return true; }
    
    virtual std::shared_ptr<Material> getMaterial(const Vec& p) const {
        return includes(p) ? material : nullptr;
    }
    
};

}       // namespace plask

#endif
