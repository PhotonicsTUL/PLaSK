#ifndef PLASK__CALCULATION_SPACE_H
#define PLASK__CALCULATION_SPACE_H

#include "transform_space_cartesian.h"
#include "transform_space_cylindric.h"
#include "border.h"

#include "../axes.h"

#include <boost/signals2.hpp>
#include "../utils/event.h"

namespace plask {

/**
 * Base class for all geometry trunks. Solvers can do calculation in calculation space with specific type.
 *
 * Almost all GeometryElement methods are delegate to child of this.
 */
struct Geometry: public GeometryElement {

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
    Geometry(shared_ptr<Material> defaultMaterial = make_shared<Air>()): defaultMaterial(defaultMaterial) {}

    /**
     * Initialize this to be the same as @p to_copy but doesn't have any changes observer.
     * @param to_copy object to copy
     */
    Geometry(const Geometry& to_copy): defaultMaterial(to_copy.defaultMaterial) {}

    /**
     * Set this to be the same as @p to_copy but doesn't changed changes observer.
     * @param to_copy object to copy
     */
    Geometry& operator=(const Geometry& to_copy) { defaultMaterial = to_copy.defaultMaterial; return *this; }

    /// Inform observators that this is deleting.
    virtual ~Geometry() { fireChanged(Event::DELETE); }

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
     * Set borders using string value which is gotten from @p borderValuesGetter.
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

    virtual Type getType() const { return TYPE_GEOMETRY; }

    /**
     * Get 3D element held by this geometry (which has type Extrusion or Revolution for 2d geometries).
     * @param 3D geometry element held by this geometry
     */
    virtual shared_ptr< GeometryElementD<3> > getElement3D() const = 0;

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

    static const std::map<std::string, std::string> null_borders;
};


/**
 * Base class for all geometry trunks in given space.
 * @tparam dim number of speace dimensions
 */
template <int dim>
class GeometryD: public Geometry {

    /// Connection object with child. It is necessary since disconnectOnChileChanged doesn't work
    boost::signals2::connection connection_with_child;

  public:

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
     * Refresh bounding box cache. Called by childrenChanged signal. Delegate this signal.
     * @param evt
     */
    void onChildChanged(const GeometryElement::Event& evt) {
        if (evt.isResize()) cachedBoundingBox = getChild()->getBoundingBox();
        //comipler should optimized out dim == 2 condition checking
        fireChanged(dim == 2 ? evt.flagsForParentWithChildrenWasChangedInformation() : evt.flagsForParent());
    }

    /// Disconnect onChildChanged from current child change signal
    void disconnectOnChildChanged() {
        //if (getChild())
        connection_with_child.disconnect();
    }

    /**
     * Initialize bounding box cache and onChange connection.
     *
     * Subclasses should call this from it's constructors (can't be moved to constructor because it uses virtual method getChild)
     * and after changing child.
     */
    void initNewChild() {
        disconnectOnChildChanged(); //disconnect old child, if any
        auto c3d = getElement3D();
        if (c3d) {
            if (c3d) connection_with_child = c3d->changedConnectMethod(this, &GeometryD<dim>::onChildChanged);
            auto c = getChild();
            if (c) cachedBoundingBox = c->getBoundingBox();
        }
    }

    virtual ~GeometryD() {
        disconnectOnChildChanged();
    }

public:

    virtual int getDimensionsCount() const { return DIMS; }

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

    virtual bool isInSubtree(const GeometryElement& el) const {
        return getChild()->isInSubtree(el);
    }

    virtual Subtree getPathsTo(const GeometryElement& el, const PathHints* pathHints = 0) const {
        return getChild()->getPathsTo(el, pathHints);
    }

    virtual void getElementsToVec(const Predicate& predicate, std::vector< shared_ptr<const GeometryElement> >& dest, const PathHints* path = 0) const {
        return getChild()->getElementsToVec(predicate, dest, path);
    }

    virtual std::size_t getChildrenCount() const {
        return getChild() ? 1 : 0;
    }

    virtual shared_ptr<GeometryElement> getChildAt(std::size_t child_nr) const {
        //if (!hasChild() || child_nr > 0) throw OutOfBoundException("Geometry::getChildAt", "child_nr");
        if (child_nr >= getChildrenCount()) throw OutOfBoundException("Geometry::getChildAt", "child_nr");
        return getChild();
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

    std::vector<CoordsType> getElementPositions(const GeometryElement& element, const PathHints* path=nullptr) const {
        return getChild()->getElementPositions(element, path);
    }

    std::vector<CoordsType> getElementPositions(const GeometryElement& element, const PathHints& path) const {
        return getChild()->getElementPositions(element, path);
    }

    std::vector<CoordsType> getElementPositions(const shared_ptr<const GeometryElement>& element, const PathHints* path=nullptr) const {
        return getChild()->getElementPositions(*element, path);
    }

    std::vector<CoordsType> getElementPositions(const shared_ptr<const GeometryElement>& element, const PathHints& path) const {
        return getChild()->getElementPositions(*element, path);
    }

    std::vector<typename Primitive<DIMS>::Box> getLeafsBoundingBoxes(const PathHints* path=nullptr) const {
        return getChild()->getLeafsBoundingBoxes(path);
    }

    std::vector<typename Primitive<DIMS>::Box> getLeafsBoundingBoxes(const PathHints& path) const {
        return getChild()->getLeafsBoundingBoxes(path);
    }

    std::vector<typename Primitive<DIMS>::Box> getElementBoundingBoxes(const GeometryElement& element, const PathHints* path=nullptr) const {
        return getChild()->getElementBoundingBoxes(element, path);
    }

    std::vector<typename Primitive<DIMS>::Box> getElementBoundingBoxes(const GeometryElement& element, const PathHints& path) const {
        return getChild()->getElementBoundingBoxes(element, path);
    }

    std::vector<typename Primitive<DIMS>::Box> getElementBoundingBoxes(const shared_ptr<const GeometryElement>& element, const PathHints* path=nullptr) const {
        return getChild()->getElementBoundingBoxes(*element, path);
    }

    std::vector<typename Primitive<DIMS>::Box> getElementBoundingBoxes(const shared_ptr<const GeometryElement>& element, const PathHints& path) const {
        return getChild()->getElementBoundingBoxes(*element, path);
    }

    GeometryElement::Subtree getPathsTo(const CoordsType& point) const {
        return getChild()->getPathsTo(point);
    }

    virtual shared_ptr<const GeometryElement> changedVersion(const Changer& changer, Vec<3, double>* translation = 0) const {
        return getChild()->changedVersion(changer, translation);
    }

    virtual void setPlanarBorders(const border::Strategy& border_to_set);

    /**
     * Get the sub/super-space of this one (automatically detected)
     * \param element geometry element within the geometry tree of this subspace or with this space child as its sub-tree
     * \param path hints specifying particular instance of the geometry element
     * \param copyBorders indicates wheter the new space should have the same borders as this one
     * \return new space
     */
    virtual GeometryD<DIMS>* getSubspace(const shared_ptr<GeometryElementD<dim>>& element, const PathHints* path=nullptr, bool copyBorders=false) const = 0;

    /**
     * Get the sub/super-space of this one (automatically detected) with specified borders
     * \param element geometry element within the geometry tree of this subspace or with this space child as its sub-tree
     * \param path hints specifying particular instance of the geometry element
     * \param borders map of edge name to border description
     * \param axesNames name of the axes for borders
     * \return new space
     */
    virtual GeometryD<DIMS>* getSubspace(const shared_ptr<GeometryElementD<dim>>& element, const PathHints* path=nullptr,
                                                 const std::map<std::string, std::string>& borders=null_borders,
                                                 const AxisNames& axesNames=AxisNames("lon","tran","up")) const {
        GeometryD<dim>* subspace = getSubspace(element, path, false);
        subspace->setBorders( [&](const std::string& s) -> boost::optional<std::string> {
            auto b = borders.find(s);
            return (b != borders.end()) ? boost::optional<std::string>(b->second) : boost::optional<std::string>();
        }, axesNames);
        return subspace;
    }


};

/**
 * Geometry trunk in 2D Cartesian space
 * @see plask::Extrusion
 */
class Geometry2DCartesian: public GeometryD<2> {

    shared_ptr<Extrusion> extrusion;

    border::StrategyPairHolder<Primitive<2>::DIRECTION_TRAN> leftright;
    border::StrategyPairHolder<Primitive<2>::DIRECTION_UP> bottomup;

    shared_ptr<Material> frontMaterial;
    shared_ptr<Material> backMaterial;

public:

    static constexpr const char* NAME = "cartesian" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D;

    virtual std::string getTypeName() const { return NAME; }

    /**
     * Set strategy for the left border.
     * @param newValue new strategy for the left border
     */
    void setLeftBorder(const border::Strategy& newValue) { leftright.setLo(newValue); fireChanged(Event::BORDERS); }

    /**
     * Get left border strategy.
     * @return left border strategy
     */
    const border::Strategy& getLeftBorder() { return leftright.getLo(); }

    /**
     * Set strategy for the right border.
     * @param newValue new strategy for the right border
     */
    void setRightBorder(const border::Strategy& newValue) { leftright.setHi(newValue); fireChanged(Event::BORDERS); }

    /**
     * Get right border strategy.
     * @return right border strategy
     */
    const border::Strategy& getRightBorder() { return leftright.getHi(); }

    /**
     * Set strategy for the bottom border.
     * @param newValue new strategy for the bottom border
     */
    void setBottomBorder(const border::Strategy& newValue) { bottomup.setLo(newValue); fireChanged(Event::BORDERS); }

    /**
     * Get bottom border strategy.
     * @return bottom border strategy
     */
    const border::Strategy& getBottomBorder() { return bottomup.getLo(); }

    /**
     * Set strategy for the top border.
     * @param newValue new strategy for the top border
     */
    void setTopBorder(const border::Strategy& newValue) { bottomup.setHi(newValue); fireChanged(Event::BORDERS); }

    /**
     * Get top border strategy.
     * @return top border strategy
     */
    const border::Strategy& getTopBorder() { return bottomup.getHi(); }

    /**
     * Set strategies for both borders in specified direction
     * \param direction direction of the borders
     * \param border_lo new strategy for the border with lower coordinate
     * \param border_hi new strategy for the border with higher coordinate
     */
    void setBorders(DIRECTION direction, const border::Strategy& border_lo, const border::Strategy& border_hi);

    /**
     * Set strategies for a border in specified direction
     * \param direction direction of the borders
     * \param higher indicates whether higher- or lower-coordinate border is to be set
     * \param border_to_set new strategy for the border with higher coordinate
     */
    void setBorder(DIRECTION direction, bool higher, const border::Strategy& border_to_set);

    const border::Strategy& getBorder(DIRECTION direction, bool higher) const;

    /**
     * Set material on the positive side of the axis along the extrusion.
     * \param material material to set
     */
    void setFrontMaterial(const shared_ptr<Material> material) { frontMaterial = material; fireChanged(Event::BORDERS); }

    /// \return material on the positive side of the axis along the extrusion
    shared_ptr<Material> getFrontMaterial() const { return frontMaterial ? frontMaterial : defaultMaterial; }

    /**
     * Set material on the negative side of the axis along the extrusion.
     * \param material material to set
     */
    void setBackMaterial(const shared_ptr<Material> material) { backMaterial = material; fireChanged(Event::BORDERS); }

    /// \return material on the negative side of the axis along extrusion
    shared_ptr<Material> getBackMaterial() const { return backMaterial ? backMaterial : defaultMaterial; }

    /**
     * Construct geometry over given @p extrusion element.
     * @param extrusion extrusion geometry element
     */
    Geometry2DCartesian(shared_ptr<Extrusion> extrusion = shared_ptr<Extrusion>());

    /**
     * Construct geometry over extrusion element build on top of given 2D @p childGeometry and with given @p length.
     *
     * It construct new extrusion element internally.
     * @param childGeometry, length parameters which will be passed to plask::Extrusion constructor
     */
    Geometry2DCartesian(shared_ptr<GeometryElementD<2>> childGeometry, double length);

    /**
     * Get child of extrusion element used by this geometry.
     * @return child geometry
     */
    virtual shared_ptr< GeometryElementD<2> > getChild() const;

    void removeAtUnsafe(std::size_t) { extrusion->setChildUnsafe(shared_ptr< GeometryElementD<2> >()); }

    virtual shared_ptr<Material> getMaterial(const Vec<2, double>& p) const;

    /**
     * Get extrusion element included in this geometry.
     * @return extrusion element included in this geometry
     */
    shared_ptr<Extrusion> getExtrusion() const { return extrusion; }

    /**
     * Get extrusion element included in this geometry.
     * @return extrusion element included in this geometry
     */
    virtual shared_ptr< GeometryElementD<3> > getElement3D() const { return extrusion; }

    /**
     * Set new extrusion element for this geometry and inform observers about changing of geometry.
     * @param extrusion new extrusion element to set and use
     */
    void setExtrusion(shared_ptr<Extrusion> extrusion);

    virtual Geometry2DCartesian* getSubspace(const shared_ptr<GeometryElementD<2>>& element, const PathHints* path = 0, bool copyBorders = false) const;

    virtual Geometry2DCartesian* getSubspace(const shared_ptr<GeometryElementD<2>>& element, const PathHints* path=nullptr,
                                          const std::map<std::string, std::string>& borders=null_borders,
                                          const AxisNames& axesNames=AxisNames()) const {
        return (Geometry2DCartesian*)GeometryD<2>::getSubspace(element, path, borders, axesNames);
    }

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_element, const AxisNames& axes) const;

    virtual void writeXML(XMLWriter::Element& parent_xml_element, WriteXMLCallback& write_cb, AxisNames axes) const;

};

/**
 * Geometry trunk in 2D Cylindrical space
 * @see plask::Revolution
 */
class Geometry2DCylindrical: public GeometryD<2> {

    shared_ptr<Revolution> revolution;

    border::StrategyPairHolder<Primitive<2>::DIRECTION_TRAN, border::UniversalStrategy> innerouter;
    border::StrategyPairHolder<Primitive<2>::DIRECTION_UP> bottomup;

    static void ensureBoundDirIsProper(DIRECTION direction, bool hi) {
        Primitive<3>::ensureIsValid2DDirection(direction);
    }

public:

    static constexpr const char* NAME = "cylindrical";

    virtual std::string getTypeName() const { return NAME; }

    /**
     * Set strategy for inner border.
     * @param newValue new strategy for inner border
     */
    void setInnerBorder(const border::UniversalStrategy& newValue) { innerouter.setLo(newValue); fireChanged(Event::BORDERS); }

    /**
     * Get inner border strategy.
     * @return inner border strategy
     */
    const border::UniversalStrategy& getInnerBorder() { return innerouter.getLo(); }

    /**
     * Set strategy for outer border.
     * @param newValue new strategy for outer border
     */
    void setOuterBorder(const border::UniversalStrategy& newValue) { innerouter.setHi(newValue); fireChanged(Event::BORDERS); }

    /**
     * Get outer border strategy.
     * @return outer border strategy
     */
    const border::UniversalStrategy& getOuterBorder() { return innerouter.getHi(); }

    /**
     * Set strategy for bottom border.
     * @param newValue new strategy for bottom border
     */
    void setBottomBorder(const border::Strategy& newValue) { bottomup.setLo(newValue); fireChanged(Event::BORDERS); }

    /**
     * Get bottom border strategy.
     * @return bottom border strategy
     */
    const border::Strategy& getBottomBorder() { return bottomup.getLo(); }

    /**
     * Set strategy for up border.
     * @param newValue new strategy for up border
     */
    void setUpBorder(const border::Strategy& newValue) { bottomup.setHi(newValue); fireChanged(Event::BORDERS); }

    /**
     * Get up border strategy.
     * @return up border strategy
     */
    const border::Strategy& getUpBorder() { return bottomup.getHi(); }

    /**
     * Construct geometry over given @p revolution element.
     * @param revolution revolution element
     */
    Geometry2DCylindrical(shared_ptr<Revolution> revolution = shared_ptr<Revolution>());

    /**
     * Construct geometry over revolution element build on top of given 2D @p childGeometry.
     *
     * It construct new plask::Revolution element internally.
     * @param childGeometry parameters which will be passed to plask::Revolution constructor
     */
    Geometry2DCylindrical(shared_ptr<GeometryElementD<2>> childGeometry);

    /**
     * Get child of revolution element used by this geometry.
     * @return child geometry
     */
    virtual shared_ptr< GeometryElementD<2> > getChild() const;

    void removeAtUnsafe(std::size_t) { revolution->setChildUnsafe(shared_ptr< GeometryElementD<2> >()); }

    virtual shared_ptr<Material> getMaterial(const Vec<2, double>& p) const;

    /**
     * Get revolution element included in this geometry.
     * @return revolution element included in this geometry
     */
    shared_ptr<Revolution> getRevolution() const { return revolution; }

    /**
     * Get revolution element included in this geometry.
     * @return revolution element included in this geometry
     */
    virtual shared_ptr< GeometryElementD<3> > getElement3D() const { return revolution; }

    /**
     * Set new revolution element for this geometry and inform observers about changing of geometry.
     * @param revolution new revolution element to set and use
     */
    void setRevolution(shared_ptr<Revolution> revolution);

    virtual Geometry2DCylindrical* getSubspace(const shared_ptr<GeometryElementD<2>>& element, const PathHints* path = 0, bool copyBorders = false) const;

    virtual Geometry2DCylindrical* getSubspace(const shared_ptr<GeometryElementD<2>>& element, const PathHints* path=nullptr,
                                            const std::map<std::string, std::string>& borders=null_borders,
                                            const AxisNames& axesNames=AxisNames()) const {
        return (Geometry2DCylindrical*)GeometryD<2>::getSubspace(element, path, borders, axesNames);
    }

    void setBorders(DIRECTION direction, const border::Strategy& border_lo, const border::Strategy& border_hi);

    void setBorders(DIRECTION direction, const border::Strategy& border_to_set);

    void setBorder(DIRECTION direction, bool higher, const border::Strategy& border_to_set);

    const border::Strategy& getBorder(DIRECTION direction, bool higher) const;

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_element, const AxisNames& axes) const;

    void writeXML(XMLWriter::Element& parent_xml_element, WriteXMLCallback& write_cb, AxisNames axes) const;

  protected:

    virtual const char* alternativeDirectionName(std::size_t ax, std::size_t orient) {
        const char* directions[3][2] = { {"back", "front"}, {"inner", "outer"}, {"bottom", "top"} };
        return directions[ax][orient];
    }

};

/**
 * Geometry trunk in 3D space
 */
class Geometry3D: public GeometryD<3> {

    shared_ptr< GeometryElementD<3> > child;

    border::StrategyPairHolder<Primitive<3>::DIRECTION_LON> backfront;
    border::StrategyPairHolder<Primitive<3>::DIRECTION_TRAN> leftright;
    border::StrategyPairHolder<Primitive<3>::DIRECTION_UP> bottomup;

public:

    static constexpr const char* NAME = "cartesian" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D;

    virtual std::string getTypeName() const { return NAME; }

    /**
     * Set strategy for the left border.
     * @param newValue new strategy for the left border
     */
    void setLeftBorder(const border::Strategy& newValue) { leftright.setLo(newValue); fireChanged(Event::BORDERS); }

    /**
     * Get left border strategy.
     * @return left border strategy
     */
    const border::Strategy& getLeftBorder() { return leftright.getLo(); }

    /**
     * Set strategy for the right border.
     * @param newValue new strategy for the right border
     */
    void setRightBorder(const border::Strategy& newValue) { leftright.setHi(newValue); fireChanged(Event::BORDERS); }

    /**
     * Get right border strategy.
     * @return right border strategy
     */
    const border::Strategy& getRightBorder() { return leftright.getHi(); }

    /**
     * Set strategy for the bottom border.
     * @param newValue new strategy for the bottom border
     */
    void setBottomBorder(const border::Strategy& newValue) { bottomup.setLo(newValue); fireChanged(Event::BORDERS); }

    /**
     * Get bottom border strategy.
     * @return bottom border strategy
     */
    const border::Strategy& getBottomBorder() { return bottomup.getLo(); }

    /**
     * Set strategy for the top border.
     * @param newValue new strategy for the top border
     */
    void setTopBorder(const border::Strategy& newValue) { bottomup.setHi(newValue); fireChanged(Event::BORDERS); }

    /**
     * Get top border strategy.
     * @return top border strategy
     */
    const border::Strategy& getTopBorder() { return bottomup.getHi(); }

    /**
     * Set strategies for both borders in specified direction
     * \param direction direction of the borders
     * \param border_lo new strategy for the border with lower coordinate
     * \param border_hi new strategy for the border with higher coordinate
     */
    void setBorders(DIRECTION direction, const border::Strategy& border_lo, const border::Strategy& border_hi);

    void setBorders(DIRECTION direction, const border::Strategy& border_to_set);

    /**
     * Set strategies for a border in specified direction
     * \param direction direction of the borders
     * \param higher indicates whether higher- or lower-coordinate border is to be set
     * \param border_to_set new strategy for the border with higher coordinate
     */
    void setBorder(DIRECTION direction, bool higher, const border::Strategy& border_to_set);

    const border::Strategy& getBorder(DIRECTION direction, bool higher) const;

    /**
     * Construct geometry over given 3D @p child element.
     * @param child child, of equal to nullptr (default) you should call setChild before use this geometry
     */
    Geometry3D(shared_ptr<GeometryElementD<3>> child = shared_ptr<GeometryElementD<3>>());

    /**
     * Get child element used by this geometry.
     * @return child element
     */
    virtual shared_ptr< GeometryElementD<3> > getChild() const;

    /**
     * Set new child.
     * This method doesn't inform observers about change.
     * @param child new child
     */
    void setChildUnsafe(shared_ptr< GeometryElementD<3> > child) {
        if (child == this->child) return;
        this->child = child;
        this->initNewChild();
    }

    /**
     * Set new child. Informs observers about change.
     * @param child new child
     */
    void setChild(shared_ptr< GeometryElementD<3> > child) {
        //this->ensureCanHaveAsChild(*child);
        setChildUnsafe(child);
        fireChildrenChanged();
    }

    /**
     * @return @c true only if child is set (is not @c nullptr)
     */
    bool hasChild() const { return this->child != nullptr; }

    void removeAtUnsafe(std::size_t) { setChildUnsafe(shared_ptr< GeometryElementD<3> >()); }

    /**
     * Get child element used by this geometry.
     * @return child element
     */
    virtual shared_ptr< GeometryElementD<3> > getElement3D() const;

    virtual shared_ptr<Material> getMaterial(const Vec<3, double>& p) const;

    virtual Geometry3D* getSubspace(const shared_ptr<GeometryElementD<3>>& element, const PathHints* path=nullptr, bool copyBorders=false) const;

    virtual void writeXMLAttr(XMLWriter::Element& dest_xml_element, const AxisNames& axes) const;
};


}   // namespace plask

#endif // PLASK__CALCULATION_SPACE_H
