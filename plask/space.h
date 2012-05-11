#ifndef PLASK__CALCULATION_SPACE_H
#define PLASK__CALCULATION_SPACE_H

#include "geometry/transform_space_cartesian.h"
#include "geometry/transform_space_cylindric.h"
#include "geometry/border.h"

#include "axes.h"

namespace plask {

/**
 * Base class for calculation spaces. Modules can do calculation in calculation space with specific type.
 *
 * Typically, calculation space classes wrap geometry element with specific type.
 */
struct CalculationSpace {

    /// Default material (which will be used for places in which geometry doesn't define any material), typically air.
    shared_ptr<Material> defaultMaterial;

    enum DIRECTION {
        DIRECTION_LON = Primitive<3>::DIRECTION_LON,
        DIRECTION_TRAN = Primitive<3>::DIRECTION_TRAN,
        DIRECTION_UP = Primitive<3>::DIRECTION_UP
    };

    /**
     * Calculation space constructor, set default material.
     * @param defaultMaterial material which will be used for places in which geometry doesn't define any material, air by default
     */
    CalculationSpace(shared_ptr<Material> defaultMaterial = make_shared<Air>()): defaultMaterial(defaultMaterial) {}

    /**
     * Set all borders in given direction or throw exception if this borders can't be set for this calculation space or direction.
     * @param direction see DIRECTION
     * @param border_lo new border strategy for lower border in given @p direction
     * @param border_hi new border strategy for higher border in given @p direction
     */
    virtual void setBorders(DIRECTION direction, const border::Strategy& border_lo, const border::Strategy& border_hi) = 0;

    /**
     * Set all borders in given direction or throw exception if this borders can't be set for this calculation space or direction.
     * @param direction see DIRECTION
     * @param higher @c true for higher bound, @c false for lower
     * @param border_to_set new border strategy for given borders
     */
    virtual void setBorders(DIRECTION direction, const border::Strategy& border_to_set) {
        setBorders(direction, border_to_set, border_to_set);
    }

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
        setBorders(DIRECTION_UP, border_to_set);
    }

    /**
     * Set border or throw exception if this border can't be set for this calculation space or direction.
     * @param direction see DIRECTION
     * @param higher @c true for higher bound, @c false for lower
     * @param border_to_set new border strategy for given border
     */
    virtual void setBorder(DIRECTION direction, bool higher, const border::Strategy& border_to_set) = 0;

    //void setBorders(const std::function< std::unique_ptr<border::Strategy> >(const std::string& s)>& borderValuesGetter, const AxisNames& axesNames);

    /**
     * Set borders using string value which get from @p borderValuesGetter.
     * @param borderValuesGetter optionaly return border strategy string for direction(s) given in argument,
     *   argument can be one of: "borders", "planar", "<axis_name>", "<axis_name>-lo", "<axis_name>-hi"
     * @param axesNames name of axes, use to create arguments for @p borderValuesGetter
     */
    void setBorders(const std::function<boost::optional<std::string>(const std::string& s)>& borderValuesGetter, const AxisNames& axesNames);

    /**
     * Get border strategy or throw exception if border can't be get for this calculation space or direction.
     * @param direction see DIRECTION
     * @param higher @c true for higher bound, @c false for lower
     * @return border strategy for given border
     */
    virtual const border::Strategy& getBorder(DIRECTION direction, bool higher) const = 0;

    /**
     * Check if structure in given direction is symmetric, i.e. one of border in this direction is mirror.
     * @param direction direction to check
     * @return @c true only if structure is symmetric in given @p direction
     */
    bool isSymmetric(DIRECTION direction) const {
        return getBorder(direction, false).type() == border::Strategy::MIRROR || getBorder(direction, true).type() == border::Strategy::MIRROR;
    }

    /**
     * Check if structure in given direction is periodic, i.e. two borders in this direction are periodic.
     * @param direction direction to check
     * @return @c true only if structure is periodic in given @p direction
     */
    bool isPeriodic(DIRECTION direction) const {
        return getBorder(direction, false).type() == border::Strategy::PERIODIC && getBorder(direction, true).type() == border::Strategy::PERIODIC;
    }

protected:

    /**
     * Dynamic cast border to given type and throw exception in case of bad cast.
     * @param strategy border strategy to cast
     */
    template <typename BorderType>
    static const BorderType& castBorder(const border::Strategy& strategy) {
        return dynamic_cast<const BorderType&>(strategy);
    }

    /// \return alternative direction name
    /// \param ax axis
    /// \param orient orientation
    virtual const char* alternativeDirectionName(std::size_t ax, std::size_t orient) {
        const char* directions[3][2] = { {"back", "front"}, {"left", "right"}, {"bottom", "top"} };
        return directions[ax][orient];
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

    std::vector<shared_ptr<const GeometryElement>> getLeafs(const PathHints& path) const {
        return getChild()->getLeafs(path);
    }

    std::vector<CoordsType> getLeafsPositions(const PathHints* path=nullptr) const {
        return getChild()->getLeafsPositions(path);
    }

    std::vector<CoordsType> getLeafsPositions(const PathHints& path) const {
        return getChild()->getLeafsPositions(path);
    }

    std::vector<typename Primitive<DIMS>::Box> getLeafsBoundingBoxes(const PathHints* path=nullptr) const {
        return getChild()->getLeafsBoundingBoxes(path);
    }

    std::vector<typename Primitive<DIMS>::Box> getLeafsBoundingBoxes(const PathHints& path) const {
        return getChild()->getLeafsBoundingBoxes(path);
    }


    virtual void setPlanarBorders(const border::Strategy& border_to_set);

    /**
     * Get the sub/super-space of this one (automatically detected)
     * \param element geometry element within the geometry tree of this subspace or with this space child as its sub-tree
     * \param path hints specifying particular instance of the geometry element
     * \param copyBorders indicates wheter the new space should have the same borders as this one
     * \return new space
     */
    virtual CalculationSpaceD<DIMS>* getSubspace(const shared_ptr<GeometryElementD<dim>>& element, const PathHints* path=nullptr, bool copyBorders=false) const = 0;

    /**
     * Get the sub/super-space of this one (automatically detected) with specified borders
     * \param element geometry element within the geometry tree of this subspace or with this space child as its sub-tree
     * \param path hints specifying particular instance of the geometry element
     * \param borders map of edge name to border description
     * \param axesNames name of the axes for borders
     * \return new space
     */
    virtual CalculationSpaceD<DIMS>* getSubspace(const shared_ptr<GeometryElementD<dim>>& element, const PathHints* path=nullptr,
                                                 const std::map<std::string, std::string>& borders=std::map<std::string, std::string>(),
                                                 const AxisNames& axesNames="lon,tran,up") const {
        CalculationSpaceD<dim>* subspace = getSubspace(element, path, false);
        subspace->setBorders( [&](const std::string& s){
            auto b = borders.find(s);
            return (b != borders.end()) ? boost::optional<std::string>(b->second) : boost::optional<std::string>();
        }, axesNames);
        return subspace;
    }


};

/**
 * 2D calculation space over extrusion geometry.
 * @see plask::Extrusion
 */
class Space2dCartesian: public CalculationSpaceD<2> {

    shared_ptr<Extrusion> extrusion;

    border::StrategyPairHolder<Primitive<2>::DIRECTION_TRAN> leftright;
    border::StrategyPairHolder<Primitive<2>::DIRECTION_UP> bottomup;

    shared_ptr<Material> frontMaterial;
    shared_ptr<Material> backMaterial;

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

    void setBorders(DIRECTION direction, const border::Strategy& border_lo, const border::Strategy& border_hi);

    void setBorder(DIRECTION direction, bool higher, const border::Strategy& border_to_set);

    const border::Strategy& getBorder(DIRECTION direction, bool higher) const;

    /**
     * Get up border strategy.
     * @return up border strategy
     */
    const border::Strategy& getUpBorder() { return bottomup.getHi(); }

    /// Set material on the positive side of the axis along the extrusion
    /// \param material material to set
    void setFrontMaterial(const shared_ptr<Material> material) { frontMaterial = material; }

    /// \return material on the positive side of the axis along the extrusion
    shared_ptr<Material> getFrontMaterial() const { return frontMaterial ? frontMaterial : defaultMaterial; }

    /// Set material on the negative side of the axis along the extrusion
    /// \param material material to set
    void setBackMaterial(const shared_ptr<Material> material) { backMaterial = material; }

    /// \return material on the negative side of the axis along extrusion
    shared_ptr<Material> getBackMaterial() const { return backMaterial ? backMaterial : defaultMaterial; }


    Space2dCartesian(const shared_ptr<Extrusion>& extrusion);

    Space2dCartesian(const shared_ptr<GeometryElementD<2>>& childGeometry, double length);

    virtual shared_ptr< GeometryElementD<2> > getChild() const;

    virtual shared_ptr<Material> getMaterial(const Vec<2, double>& p) const;

    shared_ptr<Extrusion> getExtrusion() const { return extrusion; }

    virtual Space2dCartesian* getSubspace(const shared_ptr<GeometryElementD<2>>& element, const PathHints* path = 0, bool copyBorders = false) const;

    virtual Space2dCartesian* getSubspace(const shared_ptr<GeometryElementD<2>>& element, const PathHints* path=nullptr,
                                          const std::map<std::string, std::string>& borders=std::map<std::string, std::string>(),
                                          const AxisNames& axesNames=AxisNames()) const {
        return (Space2dCartesian*)CalculationSpaceD<2>::getSubspace(element, path, borders, axesNames);
    }


};

/**
 * 2D calculation space over revolution geometry.
 * @see plask::Revolution
 */
class Space2dCylindrical: public CalculationSpaceD<2> {

    shared_ptr<Revolution> revolution;

    border::StrategyHolder<Primitive<2>::DIRECTION_TRAN, border::UniversalStrategy> outer;
    border::StrategyPairHolder<Primitive<2>::DIRECTION_UP> bottomup;

    static void ensureBoundDirIsProper(DIRECTION direction, bool hi) {
        Primitive<3>::ensureIsValid2dDirection(direction);
        if (direction == DIRECTION_TRAN && !hi)
            throw BadInput("setBorders", "Space2dCylindrical: Lower bound is not allowed in the transverse direction.");
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

    virtual Space2dCylindrical* getSubspace(const shared_ptr<GeometryElementD<2>>& element, const PathHints* path = 0, bool copyBorders = false) const;

    virtual Space2dCylindrical* getSubspace(const shared_ptr<GeometryElementD<2>>& element, const PathHints* path=nullptr,
                                            const std::map<std::string, std::string>& borders=std::map<std::string, std::string>(),
                                            const AxisNames& axesNames=AxisNames()) const {
        return (Space2dCylindrical*)CalculationSpaceD<2>::getSubspace(element, path, borders, axesNames);
    }

    void setBorders(DIRECTION direction, const border::Strategy& border_lo, const border::Strategy& border_hi);

    void setBorders(DIRECTION direction, const border::Strategy& border_to_set);

    void setBorder(DIRECTION direction, bool higher, const border::Strategy& border_to_set);

    const border::Strategy& getBorder(DIRECTION direction, bool higher) const;

  protected:

    virtual const char* alternativeDirectionName(std::size_t ax, std::size_t orient) {
        const char* directions[3][2] = { {"back", "front"}, {"inner", "outer"}, {"bottom", "top"} };
        return directions[ax][orient];
    }

};

/**
 * 3D calculation space over 3d geometry.
 */
class Space3d: public CalculationSpaceD<3> {
};


}   // namespace plask

#endif // PLASK__CALCULATION_SPACE_H
