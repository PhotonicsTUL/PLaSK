#ifndef PLASK__CALCULATION_SPACE_H
#define PLASK__CALCULATION_SPACE_H

#include "transform_space_cartesian.h"
#include "transform_space_cylindric.h"
#include "border.h"

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

    /// Childs bounding box
    typename Primitive<dim>::Box cachedBoundingBox;

    /**
     * Refresh bounding box cache. Called by children changed signal.
     * @param evt
     */
    void onChildChanged(const GeometryElement::Event& evt) {
        if (evt.isResize()) cachedBoundingBox = getChild()->getBoundingBox();
    }

    /**
     * Initialize bounding box cache.
     * Subclasses should call this from it's constructors (can't be moved to constructor because use virtual method getChild).
     */
    void init() {
        getChild()->changedConnectMethod(this, &CalculationSpaceD<dim>::onChildChanged);
        cachedBoundingBox = getChild()->getBoundingBox();
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

    /**
     * Get bounding box of child geometry.
     * @return bounding box of child geometry
     */
    const typename Primitive<dim>::Box & getChildBoundingBox() const {
        return cachedBoundingBox;
    }

    std::vector< shared_ptr<const GeometryElement> > getLeafs() const {
        return getChild()->getLeafs();
    }

};

/**
 * 2d calculation space over extrusion geometry.
 * @see plask::Extrusion
 */
class Space2DCartesian: public CalculationSpaceD<2> {

    shared_ptr<Extrusion> extrusion;

public:

    border::StrategyHolder<Primitive<2>::DIRECTION_TRAN> left, right;
    border::StrategyHolder<Primitive<2>::DIRECTION_UP> up, bottom;

    Space2DCartesian(const shared_ptr<Extrusion>& extrusion);

    Space2DCartesian(const shared_ptr<GeometryElementD<2>>& childGeometry, double length);

    virtual shared_ptr< GeometryElementD<2> > getChild() const;

    virtual shared_ptr<Material> getMaterial(const Vec<2, double>& p) const;

};

/**
 * 2d calculation space over revolution geometry.
 * @see plask::Revolution
 */
class Space2DCylindrical: public CalculationSpaceD<2> {

    shared_ptr<Revolution> revolution;

public:

    border::StrategyHolder<Primitive<2>::DIRECTION_TRAN, border::UniversalStrategy> right;
    border::StrategyHolder<Primitive<2>::DIRECTION_UP> up, bottom;

    Space2DCylindrical(const shared_ptr<Revolution>& revolution);

    Space2DCylindrical(const shared_ptr<GeometryElementD<2>>& childGeometry);

    virtual shared_ptr< GeometryElementD<2> > getChild() const;

    virtual shared_ptr<Material> getMaterial(const Vec<2, double>& p) const;

};

}   // namespace plask

#endif // PLASK__CALCULATION_SPACE_H
