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
#ifndef PLASK__GEOMETRY_CIRCLE_H
#define PLASK__GEOMETRY_CIRCLE_H

/** @file
This file contains circle (geometry object) class.
*/

#include "leaf.hpp"

namespace plask {

/**
 * Represents circle (sphere in 3D) with given radius and center at point (0, 0).
 * @ingroup GEOMETRY_OBJ
 */
template <int dim> struct PLASK_API Circle : public GeometryObjectLeaf<dim> {
    double radius;  ///< radius of this circle

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<dim>::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<dim>::Box Box;

    static const char* NAME;

    std::string getTypeName() const override;

    explicit Circle(double radius, const shared_ptr<Material>& material = shared_ptr<Material>());

    explicit Circle(double radius, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
        : GeometryObjectLeaf<dim>(materialTopBottom), radius(radius) {
        if (radius < 0.) radius = 0.;
    }

    explicit Circle(const Circle& src) : GeometryObjectLeaf<dim>(src), radius(src.radius) {}

    shared_ptr<GeometryObject> shallowCopy() const override { return make_shared<Circle>(*this); }

    Box getBoundingBox() const override;

    bool contains(const DVec& p) const override;

    void addPointsAlongToSet(std::set<double>& points,
                        Primitive<3>::Direction direction,
                        unsigned max_steps,
                        double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<dim>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

    /**
     * Set radius and inform observers about changes.
     * @param radius new radius
     */
    void setRadius(double new_radius) {
        if (new_radius < 0.) new_radius = 0.;
        this->radius = new_radius;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }
};

template <> typename Circle<2>::Box Circle<2>::getBoundingBox() const;
template <> typename Circle<3>::Box Circle<3>::getBoundingBox() const;

template <>
void Circle<2>::addLineSegmentsToSet(std::set<typename GeometryObjectD<2>::LineSegment>& segments,
                                     unsigned max_steps,
                                     double min_step_size) const;

template <>
void Circle<3>::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                     unsigned max_steps,
                                     double min_step_size) const;

PLASK_API_EXTERN_TEMPLATE_STRUCT(Circle<2>)
PLASK_API_EXTERN_TEMPLATE_STRUCT(Circle<3>)

}  // namespace plask

#endif  // PLASK__GEOMETRY_CIRCLE_H
