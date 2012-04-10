#ifndef PLASK__CALCULATION_SPACE_H
#define PLASK__CALCULATION_SPACE_H

#include "transform_space_cartesian.h"
#include "transform_space_cylindric.h"

namespace plask {

/**
 * Base class for calculation spaces. Modules can do calculation in calculation space with specific type.
 *
 * Typically, calculation space classes wrap geometry element with specific type.
 */
struct CalculationSpace {

    /// Default material, typically air.
    shared_ptr<Material> defaultMaterial;

    CalculationSpace(): defaultMaterial(make_shared<Air>()) {}

};


/**
 * Base class for calculation spaces in given space.
 * @tparam dim number of speace dimensions
 */
template <int dim>
class CalculationSpaceD: public CalculationSpace {

protected:

    /**
     * Get material from geometry or return default material if geometry returns nullptr.
     * @return default material in each point for which geometry return nullptr or material from geometry
     */
    shared_ptr<Material> getMaterialOrDefault(const Vec<dim, double>& p) const {
        auto real_mat = getChild()->getMaterial(p);
        return real_mat ? real_mat : defaultMaterial;
    }

public:

    /**
     * Get material in point @p p of child space.
     *
     * Material is getted from geometry (if geometry define material in given point) or enviroment (in another cases).
     * Result is defined, and is not nullptr, for each point @p p.
     *
     * Default implementaion just call getMaterialOrDefault which returns default material in each point for which geometry return nullptr.
     * For other stategies see subclasses of this class.
     * @param p point
     * @return material, which is not nullptr
     */
    virtual shared_ptr<Material> getMaterial(const Vec<dim, double>& p) const {
        return getMaterialOrDefault(p);
    }

    /**
     * Get child geometry.
     * @return child geometry
     */
    virtual shared_ptr< GeometryElementD<dim> > getChild() const = 0;

};

/**
 * 2d calculation space over extrusion geometry.
 * @see plask::Extrusion
 */
class Space2DCartesian: public CalculationSpaceD<2> {

    shared_ptr<Extrusion> extrusion;

public:

    Space2DCartesian(const shared_ptr<Extrusion>& extrusion): extrusion(extrusion) {}

    Space2DCartesian(const shared_ptr<GeometryElementD<2>>& childGeometry, double length)
        : extrusion(make_shared<Extrusion>(childGeometry, length)) {}

    virtual shared_ptr< GeometryElementD<2> > getChild() const;

};

struct Space2DCartesianDragToEdge: public Space2DCartesian {
    
    /// Type of extensions for infinite stacks
    enum ExtendType {
        EXTEND_NONE = 0,
        EXTEND_VERTICAL = 1,
        EXTEND_TRAN = 2,
        EXTEND_LON = 4,
        EXTEND_HORIZONTAL = 6,
        EXTEND_ALL = 7
    };
    
private:

    ExtendType _extend;
    
    Box2d cachedBoundingBox;
    
    void onChildChanged(const GeometryElement::Event& evt);

public:
    
    Space2DCartesianDragToEdge(const shared_ptr<Extrusion>& extrusion, ExtendType extendType);
    
    /** Set the extend
     * @param extend new extend
     */
    inline void setExtend(ExtendType extend) {
        if (extend == EXTEND_LON) throw BadInput("Background<2>", "EXTEND_LON not allowed for 2D background.");
        _extend = extend;
    }
    
    virtual shared_ptr<Material> getMaterial(const Vec<2, double>& p) const;
    
};

}   // namespace plask

#endif // PLASK__CALCULATION_SPACE_H
