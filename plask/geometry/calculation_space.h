#ifndef PLASK__CALCULATION_SPACE_H
#define PLASK__CALCULATION_SPACE_H

#include "space_changer_cartesian.h"

namespace plask {

/**
 * Base class for calculation spaces. Modules can do calculation in calculation space with specific type.
 *
 * Typically, calculation space classes wrap geometry element with specific type.
 */
class CalculationSpace {};  //TODO needed?

class CalculationSpaceOverExtrusion: public CalculationSpace {

    shared_ptr<Extrusion> extrusion;

public:

    CalculationSpaceOverExtrusion(const shared_ptr<Extrusion>& extrusion): extrusion(extrusion) {}

    CalculationSpaceOverExtrusion(const shared_ptr<GeometryElementD<2>>& childGeometry, double length)
        : extrusion(make_shared<Extrusion>(childGeometry, length)) {}

    /**
     * Get material in point @p p of child space.
     *
     * Material is getted from geometry (if geometry define material in given point) or enviroment (in another cases).
     * Result is defined, and is not nullptr, for each point @p p.
     *
     * Default implementaion return air in each point for which geometry return nullptr.
     * For other stategies see subclasses of this class.
     * @param p point
     * @return material, which is not nullptr
     */
    virtual shared_ptr<Material> getMaterial(const Vec<2, double>& p) const;




};

}   // namespace plask

#endif // PLASK__CALCULATION_SPACE_H
