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
#ifndef PLASK__TRANSFORM_SPACE_CARTESIAN_H
#define PLASK__TRANSFORM_SPACE_CARTESIAN_H

#include "transform.hpp"

namespace plask {

/**
 * Represent 3D geometry object which are extend of 2D object (child) in lon direction.
 * @ingroup GEOMETRY_OBJ
 */
class PLASK_API Extrusion: public GeometryObjectTransformSpace<3, 2> {

    typedef GeometryObjectTransformSpace<3, 2> BaseClass;

    double length;

  public:

    typedef BaseClass::ChildType ChildType;

    // Size of the calculation space.
    // spaceSize;

    explicit Extrusion(shared_ptr<ChildType> child, double length): BaseClass(child), length(length) {}

    explicit Extrusion(double length = 0.0/*,  spaceSize*/): length(length)/*, spaceSize(spaceSize)*/ {}

    static const char* NAME;

    std::string getTypeName() const override;

    double getLength() const { return length; }

    /**
     * Set length and inform observers.
     * @param new_length new length
     */
    void setLength(double new_length);

    bool contains(const DVec& p) const override;

    //TODO good but unused
    //virtual bool intersects(const Box& area) const;

    shared_ptr<Material> getMaterial(const DVec& p) const override;

    //virtual void getLeafsInfoToVec(std::vector<std::tuple<shared_ptr<const GeometryObject>, Box, DVec>>& dest, const PathHints* path = 0) const;

    Box fromChildCoords(const typename ChildType::Box& child_bbox) const override;

    //std::vector< plask::shared_ptr< const plask::GeometryObject > > getLeafs() const override;

    shared_ptr<GeometryObject> shallowCopy() const override;

    using GeometryObjectTransformSpace<3, 2>::getPathsTo;

    GeometryObject::Subtree getPathsAt(const DVec& point, bool all=false) const override;

    void writeXMLAttr(XMLWriter::Element &dest_xml_object, const AxisNames &axes) const override;

    void getPositionsToVec(const GeometryObject::Predicate& predicate, std::vector<DVec>& dest, const PathHints* path = 0) const override;

    // void extractToVec(const GeometryObject::Predicate &predicate, std::vector< shared_ptr<const GeometryObjectD<3> > >&dest, const PathHints *path = 0) const;

  private:
    /// @return true only if p can be inside this, false if for sure its not inside
    bool canBeInside(const DVec& p) const { return 0.0 <= p.lon() && p.lon() <= length; }

    /// @return true only if area can intersects this, false if for sure its not intersects
    bool canIntersect(const Box& area) const { return !(area.lower.lon() > length || area.upper.lon() < 0.0); }

    /**
     * Convert vector from this space (3d) to child space (2d).
     * @param p vector in space of this (3d)
     * @return @p p without lon coordinate, vector in space of child (2d)
     */
    static ChildVec childVec(const DVec& p) { return ChildVec(p.tran(), p.vert()); }

    /**
     * Convert box from this space (3d) to child space (2d).
     * @param r box in space of this (3d)
     * @return @p r box in space of child (2d)
     */
    static ChildBox childBox(const Box& r) { return ChildBox(childVec(r.lower), childVec(r.upper)); }

    /**
     * Convert vector from child space (2d) to this space (3d).
     * @param p vector in space of child (2d), without lon coordinate
     * @param lon lon coordinate
     * @return vector in space of this (3d)
     */
    static DVec parentVec(const ChildVec& p, double lon) { return DVec(lon, p.tran(), p.vert()); }

    /**
     * Convert box from child space (2d) to this space (3d).
     * @param r box in space of child (2d)
     * @return box in space of this (3d), in lon direction: from 0.0 to @p length
     */
    Box parentBox(const ChildBox& r) const { return Box(parentVec(r.lower, 0.0), parentVec(r.upper, length)); }

    void addPointsAlongToSet(std::set<double>& points,
                        Primitive<3>::Direction direction,
                        unsigned max_steps,
                        double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;
};

}   // namespace plask

#endif // TRANSFORM_SPACE_CARTESIAN_H
