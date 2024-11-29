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
#ifndef PLASK__GEOMETRY_TRIANGLE_H
#define PLASK__GEOMETRY_TRIANGLE_H

/** @file
This file contains triangle (geometry object) class.
*/

#include "leaf.hpp"

namespace plask {

/**
 * Represents triangle with one vertex at point (0, 0).
 * @ingroup GEOMETRY_OBJ
 */
struct PLASK_API Triangle : public GeometryObjectLeaf<2> {
    typedef GeometryObjectLeaf<2> BaseClass;

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename BaseClass::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename BaseClass::Box Box;

    static const char* NAME;

    std::string getTypeName() const override;

    /**
     * Construct a solid triangle with vertices at points: (0, 0), @p p0, @p p1
     * @param p0, p1 coordinates of the triangle vertices
     * @param material material inside the whole triangle
     */
    explicit Triangle(const DVec& p0 = Primitive<2>::ZERO_VEC,
                      const DVec& p1 = Primitive<2>::ZERO_VEC,
                      const shared_ptr<Material>& material = shared_ptr<Material>());

    /**
     * Construct a triangle with vertices at points: (0, 0), @p p0, @p p1
     * @param p0, p1 coordinates of the triangle vertices
     * @param materialTopBottom describes materials inside the triangle
     */
    explicit Triangle(const DVec& p0,
                      const DVec& p1,
                      shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom);

    // explicit Triangle(const DVec& p0, const DVec& p1, const std::unique_ptr<MaterialProvider>& materialProvider);

    Box2D getBoundingBox() const override;

    bool contains(const DVec& p) const override;

    shared_ptr<GeometryObject> shallowCopy() const override { return make_shared<Triangle>(*this); }

    void addPointsAlongToSet(std::set<double>& points,
                             Primitive<3>::Direction direction,
                             unsigned max_steps,
                             double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<2>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

    DVec p0, p1;

    /**
     * Set coordinates of first vertex and inform observers about changes.
     * @param new_p0 new coordinates for p0
     */
    void setP0(const DVec& new_p0) {
        p0 = new_p0;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set coordinates of second vertex and inform observers about changes.
     * @param new_p0 new coordinates for p1
     */
    void setP1(const DVec& new_p1) {
        p1 = new_p1;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }
};

}  // namespace plask

#endif  // PLASK__GEOMETRY_TRIANGLE_H
