/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#ifndef PLASK__CALCULATION_SPACE_H
#define PLASK__CALCULATION_SPACE_H

#include "edge.hpp"
#include "transform_space_cartesian.hpp"
#include "transform_space_cylindric.hpp"

#include "../axes.hpp"

#include <boost/signals2.hpp>
#include "../optional.hpp"
#include "../utils/event.hpp"

namespace plask {

/**
 * Base class for all geometry trunks. Solvers can do calculation in calculation space with specific type.
 *
 * Almost all GeometryObject methods are delegate to child of this.
 */
struct PLASK_API Geometry : public GeometryObject {
    /// Default material (which will be used for places in which geometry doesn't define any material), typically air.
    shared_ptr<Material> defaultMaterial;

    /// Axis names for this geometry
    AxisNames axisNames;

    enum Direction {
        DIRECTION_LONG = Primitive<3>::DIRECTION_LONG,
        DIRECTION_TRAN = Primitive<3>::DIRECTION_TRAN,
        DIRECTION_VERT = Primitive<3>::DIRECTION_VERT
    };

    /**
     * Calculation space constructor, set default material.
     * @param defaultMaterial material which will be used for places in which geometry doesn't define any material, air
     * by default
     */
    Geometry(shared_ptr<Material> defaultMaterial = make_shared<materials::Air>())
        : GeometryObject(PLASK_GEOMETRY_MAX_STEPS, PLASK_GEOMETRY_MIN_STEP_SIZE), defaultMaterial(defaultMaterial) {}

    /**
     * Initialize this to be the same as @p to_copy but doesn't have any changes observer.
     * @param to_copy object to copy
     */
    Geometry(const Geometry& to_copy) : GeometryObject(to_copy), defaultMaterial(to_copy.defaultMaterial) {}

    /**
     * Set this to be the same as @p to_copy but doesn't changed changes observer.
     * @param to_copy object to copy
     */
    Geometry& operator=(const Geometry& to_copy) {
        GeometryObject::operator=(to_copy);
        defaultMaterial = to_copy.defaultMaterial;
        return *this;
    }

    /// Inform observators that this is being deleted.
    virtual ~Geometry() { fireChanged(Event::EVENT_DELETE); }

    /**
     * Set all edges in given direction or throw exception if this edges can't be set for this calculation space or
     * direction.
     * @param direction see Direction
     * @param border_lo new edge strategy for lower edge in given @p direction
     * @param border_hi new edge strategy for higher edge in given @p direction
     */
    virtual void setEdges(Direction direction, const edge::Strategy& border_lo, const edge::Strategy& border_hi) = 0;

    /**
     * Set all edges in given direction or throw exception if this edges can't be set for this calculation space or
     * direction.
     * @param direction see Direction
     * @param border_to_set new edge strategy for given edges
     */
    virtual void setEdges(Direction direction, const edge::Strategy& border_to_set) {
        setEdges(direction, border_to_set, border_to_set);
    }

    /**
     * Set all planar edges or throw exception if this edges can't be set for this calculation space or direction.
     *
     * Planar edges are all edges but up-bottom.
     * @param border_to_set new edge strategy for all planar edges
     */
    virtual void setPlanarEdges(const edge::Strategy& border_to_set) = 0;

    /**
     * Set all edges (planar and up-bottom).
     * @param border_to_set new edge strategy for all edges
     */
    void setAllEdges(const edge::Strategy& border_to_set) {
        setPlanarEdges(border_to_set);
        setEdges(DIRECTION_VERT, border_to_set);
    }

    /**
     * Set edge or throw exception if this edge can't be set for this calculation space or direction.
     * @param direction see Direction
     * @param higher @c true for higher bound, @c false for lower
     * @param border_to_set new edge strategy for given edge
     */
    virtual void setEdge(Direction direction, bool higher, const edge::Strategy& border_to_set) = 0;

    // void setEdges(const std::function< std::unique_ptr<edge::Strategy> >(const std::string& s)>& borderValuesGetter,
    // const AxisNames& axesNames);

    /**
     * Set edges using string value which is gotten from @p borderValuesGetter.
     * @param borderValuesGetter optionaly return edge strategy string for direction(s) given in argument,
     *   argument can be one of: "edges", "planar", "<axis_name>", "<axis_name>-lo", "<axis_name>-hi"
     * @param axesNames name of axes, use to create arguments for @p borderValuesGetter
     * @param materialsDB source of materials
     * @param draft ignore errors
     */
    void setEdges(const std::function<plask::optional<std::string>(const std::string& s)>& borderValuesGetter,
                  const AxisNames& axesNames,
                  const MaterialsDB& materialsDB = MaterialsDB::getDefault(),
                  bool draft = false);

    /**
     * Get edge strategy or throw exception if edge can't be get for this calculation space or direction.
     * @param direction see Direction
     * @param higher @c true for higher bound, @c false for lower
     * @return edge strategy for given edge
     */
    virtual const edge::Strategy& getEdge(Direction direction, bool higher) const = 0;

    /**
     * Check if structure in given direction is symmetric, i.e. one of edge in this direction is mirror.
     * @param direction direction to check
     * @return @c true only if structure is symmetric in given @p direction
     */
    virtual bool isSymmetric(Direction direction) const {
        return getEdge(direction, false).type() == edge::Strategy::MIRROR ||
               getEdge(direction, true).type() == edge::Strategy::MIRROR;
    }

    /**
     * Check if structure in given direction is periodic, i.e. two edges in this direction are periodic.
     * @param direction direction to check
     * @return @c true only if structure is periodic in given @p direction
     */
    bool isPeriodic(Direction direction) const {
        return getEdge(direction, false).type() == edge::Strategy::PERIODIC ||
               getEdge(direction, true).type() == edge::Strategy::PERIODIC;
    }

    /**
     * Check if structure extends in given direction.
     * \param direction direction to check
     * \param higher \c true for higher bound, \c false for lower
     * \return \c true only if structure is periodic in given \p direction
     */
    bool isExtended(Direction direction, bool higher) const { return getEdge(direction, higher).type() == edge::Strategy::EXTEND; }

    Type getType() const override { return TYPE_GEOMETRY; }

    /**
     * Get 3D object held by this geometry (which has type Extrusion or Revolution for 2D geometries).
     * @return 3D geometry object held by this geometry
     */
    virtual shared_ptr<GeometryObjectD<3>> getObject3D() const = 0;

  protected:
    /**
     * Dynamic cast edge to given type and throw exception in case of bad cast.
     * @param strategy edge strategy to cast
     */
    template <typename EdgeType> static const EdgeType& castEdge(const edge::Strategy& strategy) {
        return dynamic_cast<const EdgeType&>(strategy);
    }

    /// \return alternative direction name
    /// \param ax axis
    /// \param orient orientation
    virtual const char* alternativeDirectionName(std::size_t ax, std::size_t orient) const {
        const char* directions[3][2] = {{"back", "front"}, {"left", "right"}, {"bottom", "top"}};
        return directions[ax][orient];
    }

    void storeEdgeInXML(XMLWriter::Element& dest_xml_object, Direction direction, bool higher) const;
};

/**
 * Base class for all geometry trunks in given space.
 * @tparam dim number of speace dimensions
 */
template <int dim> class PLASK_API GeometryD : public Geometry {
    /// Connection object with child. It is necessary since disconnectOnChileChanged doesn't work
    boost::signals2::connection connection_with_child;

  public:
    enum { DIM = dim };

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
    void onChildChanged(const GeometryObject::Event& evt);

    /// Disconnect onChildChanged from current child change signal
    void disconnectOnChildChanged();

    /**
     * Initialize bounding box cache and onChange connection.
     *
     * Subclasses should call this from it's constructors (can't be moved to constructor because it uses virtual method
     * getChildUnsafe) and after changing child.
     */
    void initNewChild();

    virtual ~GeometryD() { disconnectOnChildChanged(); }

  public:
    int getDimensionsCount() const override;

    /**
     * Get material in point @p p of child space.
     *
     * Material is got from the geometry (if geometry defines material at given point) or the environment (otherwise).
     * Result is defined, and is not nullptr, for each point @p p.
     *
     * Default implementation just call getMaterialOrDefault which returns default material in each point for which
     * geometry return nullptr. For other strategies see subclasses of this class.
     * @param p point
     * @return material, which is not nullptr
     */
    virtual shared_ptr<Material> getMaterial(const Vec<dim, double>& p) const;

    /**
     * Get child geometry.
     *
     * Throws excpetion if has no child.
     * @return child geometry
     */
    virtual shared_ptr<GeometryObjectD<dim>> getChild() const = 0;

    /**
     * Get child geometry.
     * @return child geometry or @c nullptr if there is no child
     */
    virtual shared_ptr<GeometryObjectD<dim>> getChildUnsafe() const = 0;

    /**
     * Get bounding box of child geometry.
     * @return bounding box of child geometry
     */
    const typename Primitive<dim>::Box& getChildBoundingBox() const { return cachedBoundingBox; }

    /**
     * Check if @a el is in subtree with child (see getChild) of @c this in root.
     * @param el object to search for
     * @return @c true only if @a el is in subtree with child of @c this in root
     */
    bool hasInSubtree(const GeometryObject& el) const override { return getChild()->hasInSubtree(el); }

    bool hasInSubtree(shared_ptr<const GeometryObject> el, const PathHints* pathHints) const {
        return getChild()->hasInSubtree(*el, pathHints);
    }

    /**
     * Find paths to @a el.
     * @param el object to search for
     * @param pathHints (optional) path hints which limits search space
     * @return sub-tree with paths to given object (@p el is in all leafs), empty sub-tree if @p el is not in subtree
     * with child of @c this in root
     */
    Subtree getPathsTo(const GeometryObject& el, const PathHints* pathHints = 0) const override {
        return getChild()->getPathsTo(el, pathHints);
    }

    /**
     * Append all objects from subtree with child (see getChild) of this in root, which fullfil predicate to vector @p
     * dest.
     * @param predicate predicate required to match
     * @param dest destination vector
     * @param path (optional) path hints which limits search space
     */
    void getObjectsToVec(const Predicate& predicate,
                         std::vector<shared_ptr<const GeometryObject>>& dest,
                         const PathHints* path = 0) const override {
        return getChild()->getObjectsToVec(predicate, dest, path);
    }

    /**
     * Get number of children of @c this.
     * @return 1 if this has a child or 0 if it hasn't
     */
    std::size_t getChildrenCount() const override { return getObject3D() ? 1 : 0; }

    /**
     * Get child_no-th child of this.
     * @param child_no must be 0
     * @return child of this
     */
    shared_ptr<GeometryObject> getChildNo(std::size_t child_no) const override {
        // if (!hasChild() || child_no > 0) throw OutOfBoundsException("Geometry::getChildNo", "child_no");
        if (child_no >= getChildrenCount()) throw OutOfBoundsException("Geometry::getChildNo", "child_no");
        return getObject3D();
    }

    /**
     * Get all leafs in subtree with child (see getChild) of this object as root.
     * @param path (optional) path hints which limits search space
     * @return all leafs in subtree with child of this object as root
     */
    std::vector<shared_ptr<const GeometryObject>> getLeafs(const PathHints* path = nullptr) const {
        return getChild()->getLeafs(path);
    }

    /**
     * Get all leafs in subtree with child (see getChild) of this object as root.
     * @param path (optional) path hints which limits search space
     * @return all leafs in subtree with child of this object as root
     */
    std::vector<shared_ptr<const GeometryObject>> getLeafs(const PathHints& path) const { return getChild()->getLeafs(path); }

    /**
     * Calculate and return a vector of positions of all leafs, optionally marked by path.
     * @param path (optional) path hints which limits search space
     * @return positions of leafs in the sub-tree with child (see getChild) of this object in the root, in the same
     * order which is generated by GeometryObject::getLeafs
     *
     * Some leafs can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     */
    std::vector<CoordsType> getLeafsPositions(const PathHints* path = nullptr) const { return getChild()->getLeafsPositions(path); }

    /**
     * Calculate and return a vector of positions of all leafs, optionally marked by path.
     * @param path path hints which limits search space
     * @return positions of leafs in the sub-tree with child (see getChild) of this object in the root, in the same
     * order which is generated by GeometryObject::getLeafs
     *
     * Some leafs can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     */
    std::vector<CoordsType> getLeafsPositions(const PathHints& path) const { return getChild()->getLeafsPositions(path); }

    /**
     * Calculate and return a vector of positions of all instances of given @p object, optionally marked by path.
     * @param object object to which instances translations should be found
     * @param path (optional) path hints which limits search space
     * @return vector of positions (relative to child of this)
     *
     * Some objects can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     */
    std::vector<CoordsType> getObjectPositions(const GeometryObject& object, const PathHints* path = nullptr) const {
        return getChild()->getObjectPositions(object, path);
    }

    /**
     * Calculate and return a vector of positions of all instances of given @p object, optionally marked by path.
     * @param object object to which instances translations should be found
     * @param path path hints which limits search space
     * @return vector of positions (relative to child of this)
     *
     * Some objects can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     */
    std::vector<CoordsType> getObjectPositions(const GeometryObject& object, const PathHints& path) const {
        return getChild()->getObjectPositions(object, path);
    }

    /**
     * Calculate and return a vector of positions of all instances of given @p object, optionally marked by path.
     * @param object object to which instances translations should be found
     * @param path (optional) path hints which limits search space
     * @return vector of positions (relative to child of this)
     *
     * Some objects can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     */
    std::vector<CoordsType> getObjectPositions(const shared_ptr<const GeometryObject>& object,
                                               const PathHints* path = nullptr) const {
        return getChild()->getObjectPositions(*object, path);
    }

    /**
     * Calculate and return a vector of positions of all instances of given @p object, optionally marked by path.
     * @param object object to which instances translations should be found
     * @param path path hints which limits search space
     * @return vector of positions (relative to child of this)
     *
     * Some objects can have vector of NaNs as translations.
     * This mean that translation is not well defined (some space changer on path).
     */
    std::vector<CoordsType> getObjectPositions(const shared_ptr<const GeometryObject>& object, const PathHints& path) const {
        return getChild()->getObjectPositions(*object, path);
    }

    /**
     * Calculate bounding boxes of all leafs, optionally marked by path.
     * @param path (optional) path hints which limits search space
     * @return bounding boxes of all leafs (relative to child of this), in the same order which is generated by
     * GeometryObject::getLeafs(const PathHints*)
     */
    std::vector<typename Primitive<DIM>::Box> getLeafsBoundingBoxes(const PathHints* path = nullptr) const {
        return getChild()->getLeafsBoundingBoxes(path);
    }

    /**
     * Calculate bounding boxes of all leafs, optionally marked by path.
     * @param path path hints which limits search space
     * @return bounding boxes of all leafs (relative to child of this), in the same order which is generated by
     * GeometryObject::getLeafs(const PathHints*)
     */
    std::vector<typename Primitive<DIM>::Box> getLeafsBoundingBoxes(const PathHints& path) const {
        return getChild()->getLeafsBoundingBoxes(path);
    }

    /**
     * Calculate bounding boxes (relative to child of this) of all instances of given \p object, optionally marked by
     * path.
     * @param object object
     * @param path (optional) path hints which limits search space
     * @return bounding boxes of all instances of given \p object
     */
    std::vector<typename Primitive<DIM>::Box> getObjectBoundingBoxes(const GeometryObject& object,
                                                                     const PathHints* path = nullptr) const {
        return getChild()->getObjectBoundingBoxes(object, path);
    }

    /**
     * Calculate bounding boxes (relative to child of this) of all instances of given \p object, optionally marked by
     * path.
     * @param object object
     * @param path path hints which limits search space
     * @return bounding boxes of all instances of given \p object
     */
    std::vector<typename Primitive<DIM>::Box> getObjectBoundingBoxes(const GeometryObject& object, const PathHints& path) const {
        return getChild()->getObjectBoundingBoxes(object, path);
    }

    /**
     * Calculate bounding boxes (relative to child of this) of all instances of given \p object, optionally marked by
     * path.
     * @param object object
     * @param path (optional) path hints which limits search space
     * @return bounding boxes of all instances of given \p object
     */
    std::vector<typename Primitive<DIM>::Box> getObjectBoundingBoxes(const shared_ptr<const GeometryObject>& object,
                                                                     const PathHints* path = nullptr) const {
        return getChild()->getObjectBoundingBoxes(*object, path);
    }

    /**
     * Calculate bounding boxes (relative to child of this) of all instances of given \p object, optionally marked by
     * path.
     * @param object object
     * @param path (optional) path hints which limits search space
     * @return bounding boxes of all instances of given \p object
     */
    std::vector<typename Primitive<DIM>::Box> getObjectBoundingBoxes(const shared_ptr<const GeometryObject>& object,
                                                                     const PathHints& path) const {
        return getChild()->getObjectBoundingBoxes(*object, path);
    }

    /**
     * Get all objects with a specified role with this object as root.
     * \param role role to search objects with
     * \return all objects with this object as root having a specified role
     */
    std::vector<shared_ptr<const GeometryObject>> getObjectsWithRole(const std::string& role) const {
        return getChild()->getObjectsWithRole(role);
    }

    /**
     * Find all paths to objects which lies at given @p point.
     * @param point point in local coordinates
     * @param all if true then return all paths if branches overlap the point
     * @return all paths, starting from child of this, last one is on top and overlies rest
     */
    GeometryObject::Subtree getPathsAt(const CoordsType& point, bool all = false) const {
        return getChild()->getPathsAt(wrapEdges(point), all);
    }

    // std::vector<shared_ptr<const GeometryObjectD<DIMS>>> extract(const Predicate& predicate, const PathHints* path =
    // 0) const {
    //     return getChild()->extract(predicate, path);
    // }

    // std::vector<shared_ptr<const GeometryObjectD<DIMS>>> extract(const Predicate& predicate, const PathHints& path)
    // const {
    //     return getChild()->extract(predicate, path);
    // }

    // std::vector<shared_ptr<const GeometryObjectD<DIMS>>> extractObject(const shared_ptr<const GeometryObjectD<DIMS>>&
    // object, const PathHints* path = 0) const {
    //     return getChild()->extractObject(*object, path);
    // }

    // std::vector<shared_ptr<const GeometryObjectD<DIMS>>> extractObject(const shared_ptr<const GeometryObjectD<DIMS>>&
    // object, const PathHints& path) const {
    //     return getChild()->extractObject(*object, path);
    // }

    /**
     * Get object closest to the root (child of this), which contains specific point and fulfills the predicate
     * \param point point to test
     * \param predicate predicate required to match, called for each object on path to point, in order from root to leaf
     * \param path optional path hints filtering out some objects
     * \return resulted object or empty pointer
     */
    inline shared_ptr<const GeometryObject> getMatchingAt(const CoordsType& point,
                                                          const Predicate& predicate,
                                                          const PathHints* path = 0) {
        return getChild()->getMatchingAt(wrapEdges(point), predicate, path);
    }

    /**
     * Get object closest to the root (child of this), which contains specific point and fulfills the predicate
     * \param point point to test
     * \param predicate predicate required to match, called for each object on path to point, in order from root to leaf
     * \param path path hints filtering out some objects
     * \return resulted object or empty pointer
     */
    inline shared_ptr<const GeometryObject> getMatchingAt(const CoordsType& point,
                                                          const Predicate& predicate,
                                                          const PathHints& path) {
        return getChild()->getMatchingAt(wrapEdges(point), predicate, path);
    }

    /**
     * Check if specified geometry object contains a point @a point.
     * \param object object to test
     * \param path path hints specifying the object
     * \param point point
     * \return true only if this geometry contains the point @a point
     */
    inline bool objectIncludes(const GeometryObject& object, const PathHints* path, const CoordsType& point) const {
        return getChild()->objectIncludes(object, path, wrapEdges(point));
    }

    /**
     * Check if specified geometry object contains a point @a point.
     * \param object object to test
     * \param path path hints specifying the object
     * \param point point
     * \return true only if this geometry contains the point @a point
     */
    inline bool objectIncludes(const GeometryObject& object, const PathHints& path, const CoordsType& point) const {
        return getChild()->objectIncludes(object, path, wrapEdges(point));
    }

    /**
     * Check if specified geometry object contains a point @a point.
     * \param object object to test
     * \param point point
     * \return true only if this geometry contains the point @a point
     */
    inline bool objectIncludes(const GeometryObject& object, const CoordsType& point) const {
        return getChild()->objectIncludes(object, wrapEdges(point));
    }

    /**
     * Check if any object at given @p point, not hidden by another object, plays role with given name @p role_name
     * (if so, returns non-nullptr).
     * @param role_name name of class
     * @param point point
     * @param path optional path hints filtering out some objects
     * @return object which is at given @p point, is not hidden by another object and plays role with name @p role_name,
     *          @c nullptr if there is not such object
     */
    inline shared_ptr<const GeometryObject> hasRoleAt(const std::string& role_name,
                                                      const CoordsType& point,
                                                      const plask::PathHints* path = 0) const {
        return getChild()->hasRoleAt(role_name, wrapEdges(point), path);
    }

    /**
     * Check if any object at given @p point, not hidden by another object, plays role with given name @p role_name
     * (if so, returns non-nullptr).
     * @param role_name name of class
     * @param point point
     * @param path optional path hints filtering out some objects
     * @return object which is at given @p point, is not hidden by another object and plays role with name @p role_name,
     *          nullptr if there is not such object
     */
    shared_ptr<const GeometryObject> hasRoleAt(const std::string& role_name,
                                               const CoordsType& point,
                                               const plask::PathHints& path) const {
        return getChild()->hasRoleAt(role_name, wrapEdges(point), path);
    }

    /**
     * Get a sum of roles sets of all objects which lies on path from this to leaf at given @p point.
     * @param point point
     * @param path path hints filtering out some objects
     * @return calculated set
     */
    std::set<std::string> getRolesAt(const CoordsType& point, const plask::PathHints* path = 0) const;

    /**
     * Get a sum of roles sets of all objects which lies on path from this to leaf at given @p point.
     * @param point point
     * @param path path hints filtering out some objects
     * @return calculated set
     */
    std::set<std::string> getRolesAt(const CoordsType& point, const plask::PathHints& path) const;

    void setPlanarEdges(const edge::Strategy& border_to_set) override;

    // /*
    //  * Get the sub/super-space of this one (automatically detected)
    //  * \param object geometry object within the geometry tree of this subspace or with this space child as its
    //  sub-tree
    //  * \param path hints specifying particular instance of the geometry object
    //  * \param copyEdges indicates wheter the new space should have the same edges as this one
    //  * \return new space
    //  */
    // virtual GeometryD<DIMS>* getSubspace(const shared_ptr<GeometryObjectD<dim>>& object, const PathHints*
    // path=nullptr, bool copyEdges=false) const = 0;

    // /*
    //  * Get the sub/super-space of this one (automatically detected) with specified edges
    //  * \param object geometry object within the geometry tree of this subspace or with this space child as its
    //  sub-tree
    //  * \param path hints specifying particular instance of the geometry object
    //  * \param edges map of edge name to edge description
    //  * \param axesNames name of the axes for edges
    //  * \return new space
    //  */
    // virtual GeometryD<DIMS>* getSubspace(const shared_ptr<GeometryObjectD<dim>>& object, const PathHints*
    // path=nullptr,
    //                                              const std::map<std::string, std::string>& edges=null_borders,
    //                                              const AxisNames& axesNames=AxisNames("lon","tran","up")) const {
    //     GeometryD<dim>* subspace = getSubspace(object, path, false);
    //     subspace->setEdges( [&](const std::string& s) -> plask::optional<std::string> {
    //         auto b = edges.find(s);
    //         return (b != edges.end()) ? plask::optional<std::string>(b->second) : plask::optional<std::string>();
    //     }, axesNames);
    //     return subspace;
    // }

    /// Wrap point to canonical position respecting edge settings (mirror, extend and periodic)
    /// \param p point to wrap
    virtual CoordsType wrapEdges(CoordsType p) const = 0;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;
};

template <> inline void GeometryD<2>::setPlanarEdges(const edge::Strategy& border_to_set) {
    setEdges(DIRECTION_TRAN, border_to_set);
}

template <> inline void GeometryD<3>::setPlanarEdges(const edge::Strategy& border_to_set) {
    setEdges(DIRECTION_LONG, border_to_set);
    setEdges(DIRECTION_TRAN, border_to_set);
}

template <> void GeometryD<2>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;
template <> void GeometryD<3>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const;

PLASK_API_EXTERN_TEMPLATE_CLASS(GeometryD<2>)
PLASK_API_EXTERN_TEMPLATE_CLASS(GeometryD<3>)

/**
 * Geometry trunk in 2D Cartesian space
 * @see plask::Extrusion
 */
class PLASK_API Geometry2DCartesian : public GeometryD<2> {
    shared_ptr<Extrusion> extrusion;

    edge::StrategyPairHolder<Primitive<2>::DIRECTION_TRAN> leftright;
    edge::StrategyPairHolder<Primitive<2>::DIRECTION_VERT> bottomup;

    shared_ptr<Material> frontMaterial;
    shared_ptr<Material> backMaterial;

  public:
    static constexpr const char* NAME = "cartesian" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D;

    std::string getTypeName() const override { return NAME; }

    /**
     * Set strategy for the left edge.
     * @param newValue new strategy for the left edge
     */
    void setLeftEdge(const edge::Strategy& newValue) {
        leftright.setLo(newValue);
        fireChanged(Event::EVENT_EDGES);
    }

    /**
     * Get left edge strategy.
     * @return left edge strategy
     */
    const edge::Strategy& getLeftEdge() { return leftright.getLo(); }

    /**
     * Set strategy for the right edge.
     * @param newValue new strategy for the right edge
     */
    void setRightEdge(const edge::Strategy& newValue) {
        leftright.setHi(newValue);
        fireChanged(Event::EVENT_EDGES);
    }

    /**
     * Get right edge strategy.
     * @return right edge strategy
     */
    const edge::Strategy& getRightEdge() { return leftright.getHi(); }

    /**
     * Set strategy for the bottom edge.
     * @param newValue new strategy for the bottom edge
     */
    void setBottomEdge(const edge::Strategy& newValue) {
        bottomup.setLo(newValue);
        fireChanged(Event::EVENT_EDGES);
    }

    /**
     * Get bottom edge strategy.
     * @return bottom edge strategy
     */
    const edge::Strategy& getBottomEdge() { return bottomup.getLo(); }

    /**
     * Set strategy for the top edge.
     * @param newValue new strategy for the top edge
     */
    void setTopEdge(const edge::Strategy& newValue) {
        bottomup.setHi(newValue);
        fireChanged(Event::EVENT_EDGES);
    }

    /**
     * Get top edge strategy.
     * @return top edge strategy
     */
    const edge::Strategy& getTopEdge() { return bottomup.getHi(); }

    /**
     * Set strategies for both edges in specified direction
     * \param direction direction of the edges
     * \param border_lo new strategy for the edge with lower coordinate
     * \param border_hi new strategy for the edge with higher coordinate
     */
    void setEdges(Direction direction, const edge::Strategy& border_lo, const edge::Strategy& border_hi) override;

    /**
     * Set strategies for a edge in specified direction
     * \param direction direction of the edges
     * \param higher indicates whether higher- or lower-coordinate edge is to be set
     * \param border_to_set new strategy for the edge with higher coordinate
     */
    void setEdge(Direction direction, bool higher, const edge::Strategy& border_to_set) override;

    const edge::Strategy& getEdge(Direction direction, bool higher) const override;

    /**
     * Set material on the positive side of the axis along the extrusion.
     * \param material material to set
     */
    void setFrontMaterial(const shared_ptr<Material> material) {
        frontMaterial = material;
        fireChanged(Event::EVENT_EDGES);
    }

    /// \return material on the positive side of the axis along the extrusion
    shared_ptr<Material> getFrontMaterial() const { return frontMaterial ? frontMaterial : defaultMaterial; }

    /**
     * Set material on the negative side of the axis along the extrusion.
     * \param material material to set
     */
    void setBackMaterial(const shared_ptr<Material> material) {
        backMaterial = material;
        fireChanged(Event::EVENT_EDGES);
    }

    /// \return material on the negative side of the axis along extrusion
    shared_ptr<Material> getBackMaterial() const { return backMaterial ? backMaterial : defaultMaterial; }

    /**
     * Construct geometry over given @p extrusion object.
     * @param extrusion extrusion geometry object
     */
    explicit Geometry2DCartesian(shared_ptr<Extrusion> extrusion = shared_ptr<Extrusion>());

    /**
     * Construct geometry over extrusion object build on top of given 2D @p childGeometry and with given @p length.
     *
     * It construct new extrusion object internally.
     * @param childGeometry, length parameters which will be passed to plask::Extrusion constructor
     */
    Geometry2DCartesian(shared_ptr<GeometryObjectD<2>> childGeometry, double length);

    /**
     * Get child of extrusion object used by this geometry.
     * @return child geometry
     */
    shared_ptr<GeometryObjectD<2>> getChild() const override;

    shared_ptr<GeometryObjectD<2>> getChildUnsafe() const override;

    void removeAtUnsafe(std::size_t) override { extrusion->setChildUnsafe(shared_ptr<GeometryObjectD<2>>()); }

    shared_ptr<Material> getMaterial(const Vec<2, double>& p) const override;

    /**
     * Get extrusion object included in this geometry.
     * @return extrusion object included in this geometry
     */
    shared_ptr<Extrusion> getExtrusion() const { return extrusion; }

    /**
     * Get extrusion object included in this geometry.
     * @return extrusion object included in this geometry
     */
    shared_ptr<GeometryObjectD<3>> getObject3D() const override { return extrusion; }

    /**
     * Set new extrusion object for this geometry and inform observers about changing of geometry.
     * @param extrusion new extrusion object to set and use
     */
    void setExtrusion(shared_ptr<Extrusion> extrusion);

    //     virtual Geometry2DCartesian* getSubspace(const shared_ptr<GeometryObjectD<2>>& object, const PathHints* path
    //     = 0, bool copyEdges = false) const;

    //     virtual Geometry2DCartesian* getSubspace(const shared_ptr<GeometryObjectD<2>>& object, const PathHints*
    //     path=nullptr,
    //                                           const std::map<std::string, std::string>& edges=null_borders,
    //                                           const AxisNames& axesNames=AxisNames()) const {
    //         return (Geometry2DCartesian*)GeometryD<2>::getSubspace(object, path, edges, axesNames);
    //     }

    CoordsType wrapEdges(CoordsType p) const override;

    shared_ptr<GeometryObject> shallowCopy() const override;

    shared_ptr<GeometryObject> deepCopy(std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const override;

    /**
     * Add characteristic points information along specified axis to set
     * \param[in,out] points ordered set of division points along specified axis
     * \param direction axis direction
     */
    void addPointsAlongToSet(std::set<double>& points, Primitive<3>::Direction direction, unsigned = 0, double = 0) const override {
        shared_ptr<GeometryObjectD<2>> child;
        try {
            child = getChild();
        } catch (const NoChildException&) {
            return;
        }
        child->addPointsAlongToSet(points, direction, max_steps, min_step_size);
    }

    /**
     * Get characteristic points information along specified axis
     * \param direction axis direction
     * \returns ordered set of division points along specified axis
     */
    std::set<double> getPointsAlong(Primitive<3>::Direction direction) const {
        std::set<double> points;
        addPointsAlongToSet(points, direction);
        return points;
    }

    /**
     * Add characteristic points to the set and edges connecting them
     * \param[in, out] segments set to extend
     */
    void addLineSegmentsToSet(std::set<typename GeometryObjectD<2>::LineSegment>& segments) const {
        shared_ptr<GeometryObjectD<2>> child;
        try {
            child = getChild();
        } catch (const NoChildException&) {
            return;
        }
        child->addLineSegmentsToSet(segments, max_steps, min_step_size);
    }

    /**
     * Add characteristic points to the set and edges connecting them
     * \return segments set
     */
    std::set<typename GeometryObjectD<2>::LineSegment> getLineSegments() const {
        std::set<typename GeometryObjectD<2>::LineSegment> segments;
        addLineSegmentsToSet(segments);
        return segments;
    }

    /**
     * Get this or copy of this child with some changes in subtree.
     * @param[in] changer changer which will be aplied to subtree with this in root
     * @param[out] translation optional, if non-null, recommended translation of this after change will be stored
     * @return pointer to this (if nothing was change) or copy of this with some changes in subtree
     */
    shared_ptr<const GeometryObject> changedVersion(const Changer& changer, Vec<3, double>* translation = 0) const override;

    void writeXML(XMLWriter::Element& parent_xml_object, WriteXMLCallback& write_cb, AxisNames axes) const override;
};

/**
 * Geometry trunk in 2D Cylindrical space
 * @see plask::Revolution
 */
class PLASK_API Geometry2DCylindrical : public GeometryD<2> {
    shared_ptr<Revolution> revolution;

    edge::StrategyPairHolder<Primitive<2>::DIRECTION_TRAN, edge::UniversalStrategy> innerouter;
    edge::StrategyPairHolder<Primitive<2>::DIRECTION_VERT> bottomup;

    static void ensureBoundDirIsProper(Direction direction /*, bool hi*/) { Primitive<3>::ensureIsValid2DDirection(direction); }

  public:
    static constexpr const char* NAME = "cylindrical";

    std::string getTypeName() const override { return NAME; }

    /**
     * Set strategy for inner edge.
     * @param newValue new strategy for inner edge
     */
    void setInnerEdge(const edge::UniversalStrategy& newValue) {
        innerouter.setLo(newValue);
        fireChanged(Event::EVENT_EDGES);
    }

    /**
     * Get inner edge strategy.
     * @return inner edge strategy
     */
    const edge::UniversalStrategy& getInnerEdge() { return innerouter.getLo(); }

    /**
     * Set strategy for outer edge.
     * @param newValue new strategy for outer edge
     */
    void setOuterEdge(const edge::UniversalStrategy& newValue) {
        innerouter.setHi(newValue);
        fireChanged(Event::EVENT_EDGES);
    }

    /**
     * Get outer edge strategy.
     * @return outer edge strategy
     */
    const edge::UniversalStrategy& getOuterEdge() { return innerouter.getHi(); }

    /**
     * Set strategy for bottom edge.
     * @param newValue new strategy for bottom edge
     */
    void setBottomEdge(const edge::Strategy& newValue) {
        bottomup.setLo(newValue);
        fireChanged(Event::EVENT_EDGES);
    }

    /**
     * Get bottom edge strategy.
     * @return bottom edge strategy
     */
    const edge::Strategy& getBottomEdge() { return bottomup.getLo(); }

    /**
     * Set strategy for up edge.
     * @param newValue new strategy for up edge
     */
    void setUpEdge(const edge::Strategy& newValue) {
        bottomup.setHi(newValue);
        fireChanged(Event::EVENT_EDGES);
    }

    /**
     * Get up edge strategy.
     * @return up edge strategy
     */
    const edge::Strategy& getUpEdge() { return bottomup.getHi(); }

    /**
     * Construct geometry over given @p revolution object.
     * @param revolution revolution object
     */
    explicit Geometry2DCylindrical(shared_ptr<Revolution> revolution = shared_ptr<Revolution>());

    /**
     * Construct geometry over revolution object build on top of given 2D @p childGeometry.
     *
     * It construct new plask::Revolution object internally.
     * @param childGeometry parameters which will be passed to plask::Revolution constructor
     */
    explicit Geometry2DCylindrical(shared_ptr<GeometryObjectD<2>> childGeometry);

    /**
     * Get child of revolution object used by this geometry.
     * @return child geometry
     */
    shared_ptr<GeometryObjectD<2>> getChild() const override;

    shared_ptr<GeometryObjectD<2>> getChildUnsafe() const override;

    void removeAtUnsafe(std::size_t) override { revolution->setChildUnsafe(shared_ptr<GeometryObjectD<2>>()); }

    shared_ptr<Material> getMaterial(const Vec<2, double>& p) const override;

    /**
     * Get revolution object included in this geometry.
     * @return revolution object included in this geometry
     */
    shared_ptr<Revolution> getRevolution() const { return revolution; }

    /**
     * Get revolution object included in this geometry.
     * @return revolution object included in this geometry
     */
    shared_ptr<GeometryObjectD<3>> getObject3D() const override { return revolution; }

    /**
     * Set new revolution object for this geometry and inform observers about changing of geometry.
     * @param revolution new revolution object to set and use
     */
    void setRevolution(shared_ptr<Revolution> revolution);

    //     virtual Geometry2DCylindrical* getSubspace(const shared_ptr<GeometryObjectD<2>>& object, const PathHints*
    //     path = 0, bool copyEdges = false) const;

    //     virtual Geometry2DCylindrical* getSubspace(const shared_ptr<GeometryObjectD<2>>& object, const PathHints*
    //     path=nullptr,
    //                                             const std::map<std::string, std::string>& edges=null_borders,
    //                                             const AxisNames& axesNames=AxisNames()) const {
    //         return (Geometry2DCylindrical*)GeometryD<2>::getSubspace(object, path, edges, axesNames);
    //     }

    void setEdges(Direction direction, const edge::Strategy& border_lo, const edge::Strategy& border_hi) override;

    void setEdges(Direction direction, const edge::Strategy& border_to_set) override;

    void setEdge(Direction direction, bool higher, const edge::Strategy& border_to_set) override;

    const edge::Strategy& getEdge(Direction direction, bool higher) const override;

    bool isSymmetric(Direction direction) const override {
        if (direction == DIRECTION_TRAN) return true;
        return getEdge(direction, false).type() == edge::Strategy::MIRROR ||
               getEdge(direction, true).type() == edge::Strategy::MIRROR;
    }

    CoordsType wrapEdges(CoordsType p) const override;

    shared_ptr<GeometryObject> shallowCopy() const override;

    shared_ptr<GeometryObject> deepCopy(std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const override;

    /**
     * Add characteristic points information along specified axis to set
     * \param[in,out] points ordered set of division points along specified axis
     * \param direction axis direction
     */
    void addPointsAlongToSet(std::set<double>& points, Primitive<3>::Direction direction, unsigned = 0, double = 0) const override {
        shared_ptr<GeometryObjectD<2>> child;
        try {
            child = getChild();
        } catch (const NoChildException&) {
            return;
        }
        child->addPointsAlongToSet(points, direction, max_steps, min_step_size);
    }

    /**
     * Get characteristic points information along specified axis
     * \param direction axis direction
     * \returns ordered set of division points along specified axis
     */
    std::set<double> getPointsAlong(Primitive<3>::Direction direction) const {
        std::set<double> points;
        addPointsAlongToSet(points, direction);
        return points;
    }

    /**
     * Add characteristic points to the set and edges connecting them
     * \param[in, out] segments set to extend
     */
    void addLineSegmentsToSet(std::set<typename GeometryObjectD<2>::LineSegment>& segments) const {
        shared_ptr<GeometryObjectD<2>> child;
        try {
            child = getChild();
        } catch (const NoChildException&) {
            return;
        }
        child->addLineSegmentsToSet(segments, max_steps, min_step_size);
    }

    /**
     * Add characteristic points to the set and edges connecting them
     * \return segments set
     */
    std::set<typename GeometryObjectD<2>::LineSegment> getLineSegments() const {
        std::set<typename GeometryObjectD<2>::LineSegment> segments;
        addLineSegmentsToSet(segments);
        return segments;
    }

    /**
     * Get this or copy of this child with some changes in subtree.
     * @param[in] changer changer which will be aplied to subtree with this in root
     * @param[out] translation optional, if non-null, recommended translation of this after change will be stored
     * @return pointer to this (if nothing was change) or copy of this with some changes in subtree
     */
    shared_ptr<const GeometryObject> changedVersion(const Changer& changer, Vec<3, double>* translation = 0) const override;

    void writeXML(XMLWriter::Element& parent_xml_object, WriteXMLCallback& write_cb, AxisNames axes) const override;

  protected:
    const char* alternativeDirectionName(std::size_t ax, std::size_t orient) const override {
        const char* directions[3][2] = {{"cw", "ccw"}, {"inner", "outer"}, {"bottom", "top"}};
        return directions[ax][orient];
    }
};

/**
 * Geometry trunk in 3D space
 */
class PLASK_API Geometry3D : public GeometryD<3> {
    shared_ptr<GeometryObjectD<3>> child;

    edge::StrategyPairHolder<Primitive<3>::DIRECTION_LONG> backfront;
    edge::StrategyPairHolder<Primitive<3>::DIRECTION_TRAN> leftright;
    edge::StrategyPairHolder<Primitive<3>::DIRECTION_VERT> bottomup;

  public:
    static constexpr const char* NAME = "cartesian" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D;

    std::string getTypeName() const override { return NAME; }

    /**
     * Set strategy for the left edge.
     * @param newValue new strategy for the left edge
     */
    void setLeftEdge(const edge::Strategy& newValue) {
        leftright.setLo(newValue);
        fireChanged(Event::EVENT_EDGES);
    }

    /**
     * Get left edge strategy.
     * @return left edge strategy
     */
    const edge::Strategy& getLeftEdge() { return leftright.getLo(); }

    /**
     * Set strategy for the right edge.
     * @param newValue new strategy for the right edge
     */
    void setRightEdge(const edge::Strategy& newValue) {
        leftright.setHi(newValue);
        fireChanged(Event::EVENT_EDGES);
    }

    /**
     * Get right edge strategy.
     * @return right edge strategy
     */
    const edge::Strategy& getRightEdge() { return leftright.getHi(); }

    /**
     * Set strategy for the bottom edge.
     * @param newValue new strategy for the bottom edge
     */
    void setBottomEdge(const edge::Strategy& newValue) {
        bottomup.setLo(newValue);
        fireChanged(Event::EVENT_EDGES);
    }

    /**
     * Get bottom edge strategy.
     * @return bottom edge strategy
     */
    const edge::Strategy& getBottomEdge() { return bottomup.getLo(); }

    /**
     * Set strategy for the top edge.
     * @param newValue new strategy for the top edge
     */
    void setTopEdge(const edge::Strategy& newValue) {
        bottomup.setHi(newValue);
        fireChanged(Event::EVENT_EDGES);
    }

    /**
     * Get top edge strategy.
     * @return top edge strategy
     */
    const edge::Strategy& getTopEdge() { return bottomup.getHi(); }

    /**
     * Set strategies for both edges in specified direction
     * \param direction direction of the edges
     * \param border_lo new strategy for the edge with lower coordinate
     * \param border_hi new strategy for the edge with higher coordinate
     */
    void setEdges(Direction direction, const edge::Strategy& border_lo, const edge::Strategy& border_hi) override;

    void setEdges(Direction direction, const edge::Strategy& border_to_set) override;

    /**
     * Set strategies for a edge in specified direction
     * \param direction direction of the edges
     * \param higher indicates whether higher- or lower-coordinate edge is to be set
     * \param border_to_set new strategy for the edge with higher coordinate
     */
    void setEdge(Direction direction, bool higher, const edge::Strategy& border_to_set) override;

    const edge::Strategy& getEdge(Direction direction, bool higher) const override;

    /**
     * Construct geometry over given 3D @p child object.
     * @param child child, of equal to nullptr (default) you should call setChild before use this geometry
     */
    explicit Geometry3D(shared_ptr<GeometryObjectD<3>> child = shared_ptr<GeometryObjectD<3>>());

    /**
     * Get child object used by this geometry.
     * @return child object
     */
    shared_ptr<GeometryObjectD<3>> getChild() const override;

    shared_ptr<GeometryObjectD<3>> getChildUnsafe() const override;

    /**
     * Set new child.
     * This method doesn't inform observers about change.
     * @param child new child
     */
    void setChildUnsafe(shared_ptr<GeometryObjectD<3>> child) {
        if (child == this->child) return;
        this->child = child;
        this->initNewChild();
    }

    /**
     * Set new child. Informs observers about change.
     * @param child new child
     */
    void setChild(shared_ptr<GeometryObjectD<3>> child) {
        // this->ensureCanHaveAsChild(*child);
        setChildUnsafe(child);
        fireChildrenChanged();
    }

    /**
     * @return @c true only if child is set (is not @c nullptr)
     */
    bool hasChild() const { return this->child != nullptr; }

    void removeAtUnsafe(std::size_t) override { setChildUnsafe(shared_ptr<GeometryObjectD<3>>()); }

    /**
     * Get child object used by this geometry.
     * @return child object
     */
    shared_ptr<GeometryObjectD<3>> getObject3D() const override;

    shared_ptr<Material> getMaterial(const Vec<3, double>& p) const override;

    CoordsType wrapEdges(CoordsType p) const override;

    shared_ptr<GeometryObject> shallowCopy() const override;

    shared_ptr<GeometryObject> deepCopy(std::map<const GeometryObject*, shared_ptr<GeometryObject>>& copied) const override;

    /**
     * Add characteristic points information along specified axis to set
     * \param[in,out] points ordered set of division points along specified axis
     * \param direction axis direction
     */
    void addPointsAlongToSet(std::set<double>& points, Primitive<3>::Direction direction, unsigned = 0, double = 0) const override {
        shared_ptr<GeometryObjectD<3>> child;
        try {
            child = getChild();
        } catch (const NoChildException&) {
            return;
        }
        child->addPointsAlongToSet(points, direction, max_steps, min_step_size);
    }

    /**
     * Get characteristic points information along specified axis
     * \param direction axis direction
     * \returns ordered set of division points along specified axis
     */
    std::set<double> getPointsAlong(Primitive<3>::Direction direction) const {
        std::set<double> points;
        addPointsAlongToSet(points, direction);
        return points;
    }

    /**
     * Add characteristic points to the set and edges connecting them
     * \param[in, out] segments set to extend
     */
    void addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments) const {
        shared_ptr<GeometryObjectD<3>> child;
        try {
            child = getChild();
        } catch (const NoChildException&) {
            return;
        }
        child->addLineSegmentsToSet(segments, max_steps, min_step_size);
    }

    /**
     * Add characteristic points to the set and edges connecting them
     * \return segments set
     */
    std::set<typename GeometryObjectD<3>::LineSegment> getLineSegments() const {
        std::set<typename GeometryObjectD<3>::LineSegment> segments;
        addLineSegmentsToSet(segments);
        return segments;
    }

    /**
     * Get this or copy of this child with some changes in subtree.
     * @param[in] changer changer which will be aplied to subtree with this in root
     * @param[out] translation optional, if non-null, recommended translation of this after change will be stored
     * @return pointer to this (if nothing was change) or copy of this with some changes in subtree
     */
    shared_ptr<const GeometryObject> changedVersion(const Changer& changer, Vec<3, double>* translation = 0) const override;

    //     virtual Geometry3D* getSubspace(const shared_ptr<GeometryObjectD<3>>& object, const PathHints* path=nullptr,
    //     bool copyEdges=false) const;
};

}  // namespace plask

#endif  // PLASK__CALCULATION_SPACE_H
