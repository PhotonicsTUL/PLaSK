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

    CalculationSpace(shared_ptr<Material> defaultMaterial = make_shared<Air>()): defaultMaterial(defaultMaterial) {}

    /**
     * Set all borders in given ddirection or throw exception if this borders can't be set for this calculation space or direction.
     * @param direction see Primitive<3>::DIRECTION
     * @param higher @c true for higher bound, @c false for lower
     * @param border_to_set new border strategy for given borders
     */
    virtual void setBorders(Primitive<3>::DIRECTION direction, const border::Strategy& border_to_set) = 0;

    /**
     * Set all planar borders or throw exception if this borders can't be set for this calculation space or direction.
     *
     * Planar borders are all borders but up-bottom.
     * @param border_to_set new border strategy for all planar borders
     */
    virtual void setPlanarBorders(const border::Strategy& border_to_set) = 0;

    /**
     * Set all borders (planar and up-bottom).
     * @param border_to_set new border strategy for all borders
     */
    void setAllBorders(const border::Strategy& border_to_set) {
        setPlanarBorders(border_to_set);
        setBorders(Primitive<3>::DIRECTION_UP, border_to_set);
    }

    /**
     * Set border or throw exception if this border can't be set for this calculation space or direction.
     * @param direction see Primitive<3>::DIRECTION
     * @param higher @c true for higher bound, @c false for lower
     * @param border_to_set new border strategy for given border
     */
    virtual void setBorder(Primitive<3>::DIRECTION direction, bool higher, const border::Strategy& border_to_set) = 0;

    /**
     * Get border strategy or throw exception if border can't be get for this calculation space or direction.
     * @param direction see Primitive<3>::DIRECTION
     * @param higher @c true for higher bound, @c false for lower
     * @return border strategy for given border
     */
    virtual const border::Strategy& getBorder(Primitive<3>::DIRECTION direction, bool higher) const = 0;


    bool isSymmetric(Primitive<3>::DIRECTION direction) const {
        return getBorder(direction, false).type() == border::Strategy::MIRROR || getBorder(direction, true).type() == border::Strategy::MIRROR;
    }

    bool isPeriodic(Primitive<3>::DIRECTION direction) const {
        return getBorder(direction, false).type() == border::Strategy::PERIODIC && getBorder(direction, true).type() == border::Strategy::PERIODIC;
    }

protected:

    /**
     * Dynamic cast border to given type and throw excpetion in case of bad cast.
     * @param strategy border strategy to cast
     */
    template <typename BorderType>
    static const BorderType& castBorder(const border::Strategy& strategy) {
        return dynamic_cast<const BorderType&>(strategy);
    }

};


/**
 * Base class for calculation spaces in given space.
 * @tparam dim number of speace dimensions
 */
template <int dim>
struct CalculationSpaceD: public CalculationSpace {

    enum  { DIMS = dim };

    typedef Vec<dim, double> CoordsType;

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
     * Refresh bounding box cache. Called by childrenChanged signal.
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
     * Material is got from geometry (if geometry define material in given point) or environment (in another cases).
     * Result is defined, and is not nullptr, for each point @p p.
     *
     * Default implementation just call getMaterialOrDefault which returns default material in each point for which geometry return nullptr.
     * For other strategies see subclasses of this class.
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
    const typename Primitive<dim>::Box& getChildBoundingBox() const {
        return cachedBoundingBox;
    }

    std::vector<shared_ptr<const GeometryElement>> getLeafs(const PathHints* path=nullptr) const {
        return getChild()->getLeafs(path);
    }

    std::vector<CoordsType> getLeafsPositions(const PathHints* path=nullptr) const {
        return getChild()->getLeafsPositions(path);
    }

    std::vector<typename Primitive<DIMS>::Box> getLeafsBoundingBoxes(const PathHints* path=nullptr) const {
        return getChild()->getLeafsBoundingBoxes(path);
    }

    virtual CalculationSpaceD<DIMS>* getSubspace(const shared_ptr< GeometryElementD<dim> >& element, const PathHints* path = 0, bool copyBorders = false) const = 0;

    void setPlanarBorders(const border::Strategy& border_to_set);
};

/**
 * 2D calculation space over extrusion geometry.
 * @see plask::Extrusion
 */
class Space2dCartesian: public CalculationSpaceD<2> {

    shared_ptr<Extrusion> extrusion;

    border::StrategyPairHolder<Primitive<2>::DIRECTION_TRAN> leftright;
    border::StrategyPairHolder<Primitive<2>::DIRECTION_UP> bottomup;

public:

    /**
     * Set strategy for left border.
     * @param newValue new strategy for left border
     */
    void setLeftBorder(const border::Strategy& newValue) { leftright.setLo(newValue); }

    /**
     * Get left border strategy.
     * @return left border strategy
     */
    const border::Strategy& getLeftBorder() { return leftright.getLo(); }

    /**
     * Set strategy for right border.
     * @param newValue new strategy for right border
     */
    void setRightBorder(const border::Strategy& newValue) { leftright.setHi(newValue); }

    /**
     * Get right border strategy.
     * @return right border strategy
     */
    const border::Strategy& getRightBorder() { return leftright.getHi(); }

    /**
     * Set strategy for bottom border.
     * @param newValue new strategy for bottom border
     */
    void setBottomBorder(const border::Strategy& newValue) { bottomup.setLo(newValue); }

    /**
     * Get bottom border strategy.
     * @return bottom border strategy
     */
    const border::Strategy& getBottomBorder() { return bottomup.getLo(); }

    /**
     * Set strategy for up border.
     * @param newValue new strategy for up border
     */
    void setUpBorder(const border::Strategy& newValue) { bottomup.setHi(newValue); }

    void setBorders(Primitive<3>::DIRECTION direction, const border::Strategy& border_to_set);

    void setBorder(Primitive<3>::DIRECTION direction, bool higher, const border::Strategy& border_to_set);

    const border::Strategy& getBorder(Primitive<3>::DIRECTION direction, bool higher) const;

    /**
     * Get up border strategy.
     * @return up border strategy
     */
    const border::Strategy& getUpBorder() { return bottomup.getHi(); }

    Space2dCartesian(const shared_ptr<Extrusion>& extrusion);

    Space2dCartesian(const shared_ptr<GeometryElementD<2>>& childGeometry, double length);

    virtual shared_ptr< GeometryElementD<2> > getChild() const;

    virtual shared_ptr<Material> getMaterial(const Vec<2, double>& p) const;

    shared_ptr<Extrusion> getExtrusion() const { return extrusion; }

    virtual Space2dCartesian* getSubspace(const shared_ptr< GeometryElementD<2> >& element, const PathHints* path = 0, bool copyBorders = false) const;

};

/**
 * 2D calculation space over revolution geometry.
 * @see plask::Revolution
 */
class Space2dCylindrical: public CalculationSpaceD<2> {

    shared_ptr<Revolution> revolution;

    border::StrategyHolder<Primitive<2>::DIRECTION_TRAN, border::UniversalStrategy> outer;
    border::StrategyPairHolder<Primitive<2>::DIRECTION_UP> bottomup;

    static void ensureBoundDirIsProper(Primitive<3>::DIRECTION direction, bool hi) {
        Primitive<3>::ensureIsValid2dDirection(direction);
        if (direction == Primitive<3>::DIRECTION_TRAN && !hi)
            throw Exception("Space2dCylindrical: Lower bound is not allowed in tran direction.");
    }

public:

    /**
     * Set strategy for outer border.
     * @param newValue new strategy for outer border
     */
    void setOuterBorder(const border::UniversalStrategy& newValue) { outer = newValue; }

    /**
     * Get outer border strategy.
     * @return outer border strategy
     */
    const border::UniversalStrategy& getOuterBorder() { return outer.getStrategy(); }

    /**
     * Set strategy for bottom border.
     * @param newValue new strategy for bottom border
     */
    void setBottomBorder(const border::Strategy& newValue) { bottomup.setLo(newValue); }

    /**
     * Get bottom border strategy.
     * @return bottom border strategy
     */
    const border::Strategy& getBottomBorder() { return bottomup.getLo(); }

    /**
     * Set strategy for up border.
     * @param newValue new strategy for up border
     */
    void setUpBorder(const border::Strategy& newValue) { bottomup.setHi(newValue); }

    /**
     * Get up border strategy.
     * @return up border strategy
     */
    const border::Strategy& getUpBorder() { return bottomup.getHi(); }

    Space2dCylindrical(const shared_ptr<Revolution>& revolution);

    Space2dCylindrical(const shared_ptr<GeometryElementD<2>>& childGeometry);

    virtual shared_ptr< GeometryElementD<2> > getChild() const;

    virtual shared_ptr<Material> getMaterial(const Vec<2, double>& p) const;

    shared_ptr<Revolution> getRevolution() const { return revolution; }

    virtual Space2dCylindrical* getSubspace(const shared_ptr< GeometryElementD<2> >& element, const PathHints* path = 0, bool copyBorders = false) const;

    void setBorders(Primitive<3>::DIRECTION direction, const border::Strategy& border_to_set);

    void setBorder(Primitive<3>::DIRECTION direction, bool higher, const border::Strategy& border_to_set);

    const border::Strategy& getBorder(Primitive<3>::DIRECTION direction, bool higher) const;

};

/**
 * 3D calculation space over 3d geometry.
 */
class Space3d: public CalculationSpaceD<3> {
};


}   // namespace plask

#endif // PLASK__CALCULATION_SPACE_H
