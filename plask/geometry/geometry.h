#ifndef PLASK__GEOMETRY_H
#define PLASK__GEOMETRY_H

#include "../material/material.h"

namespace plask {

/**
 * Base class for all geometries.
 */
struct GeometryBase {
    
    /**
     * Check if geometry is leaf.
     * @return true only if geometry is leaf.
     */
    virtual bool isLeaf() const;
    
};

template < typename PrimitivesSet >
struct Geometry: public GeometryBase {
    
    typedef typename PrimitivesSet::Rect Rect;
    typedef typename PrimitivesSet::Vec Vec;
    static const int dim = PrimitivesSet::dim;
    
    //virtual Rect getBoundingBox() const;
    
    /**
     * Check if geometry includes point.
     * @param p point
     * @return true only if this geometry includes point @a p
     */
    virtual bool includes(const Vec& p);
    
    /**
     * Check if geometry includes some point from given @a area.
     * @param area rectangular area
     * @return true only if this geometry includes some points from @a area
     */
    virtual bool includes(const Rect& area);
    
    virtual Vec getBoundingBoxSize();
    
    /**
     * @return material in given point
     */
    virtual Material* getMaterial(const Vec& p);
    
    //virtual std::vector<Material*> getMaterials(Mesh);        ??
    
};

}       // namespace plask

#endif
