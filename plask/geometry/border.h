#ifndef PLASK__GEOMETRY_BORDER_H
#define PLASK__GEOMETRY_BORDER_H

#include "../material/material.h"

namespace plask {

namespace border {

/**
 * Base, abstract for all classes which describe what do with points outside geometry in calculation space.
 */
struct Strategy {

    /**
     * Apply strategy to given point @p p.
     * @param bbox_lo[in], bbox_hi[in] coordinates of geometry element bounding box in startegy working direction
     * @param p[in,out] coordinate of point in startegy working direction, it's lower than @p bbox_lo or higher than @p bbox_hi, this method can move this point
     * @param result_material optionaly, this method can assign to it material which should be used
     */
    virtual void apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material) const = 0;

    virtual Strategy* clone() const = 0;
};

/**
 * Base class for all universal strategies.
 *
 * Universal strategies form subset of strategies, and could be required in some context.
 */
struct UniversalStrategy: public Strategy {};

/**
 * Strategy which assign constant material.
 */
struct ConstMaterial: public UniversalStrategy {

    /**
     * Material which will be assigned to result_material by apply method.
     */
    shared_ptr<Material> material;

    /**
     * Construct ConstMaterial strategy wich use given material.
     * @param material material which will be assigned to result_material by apply method
     */
    ConstMaterial(const shared_ptr<Material>& material): material(material) {}

    virtual void apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material) const;

    virtual ConstMaterial* clone() const;

};

/**
 * Strategy which move point p to nearest border.
 */
struct Extend: public UniversalStrategy {

    virtual void apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material) const;

    virtual Extend* clone() const;

};

/**
 * Strategy which move point p by multiple of (bbox_hi - bbox_lo) to be in range [bbox_lo, bbox_hi].
 */
struct Periodic: public Strategy {

    virtual void apply(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>& result_material) const;

    virtual Periodic* clone() const;
};

}   // namespace border

}   // namespace plask

#endif // PLASK__GEOMETRY_BORDER_H
