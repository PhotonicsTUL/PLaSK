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
#ifndef PLASK__GEOMETRY_ELLIPSE_H
#define PLASK__GEOMETRY_ELLIPSE_H

/** @file
This file contains ellipse (geometry object) class.
*/

#include "leaf.hpp"

namespace plask {

/**
 * Represents ellipse (sphere in 3D) with given radius and center at point (0, 0).
 * @ingroup GEOMETRY_OBJ
 */
struct PLASK_API Ellipse : public GeometryObjectLeaf<2> {
    double radius0;  ///< transverse radius of this ellipse
    double radius1;  ///< vertical radius of this ellipse

    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<2>::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<2>::Box Box;

    static const char* NAME;

    std::string getTypeName() const override;

    explicit Ellipse(double rx, double ry, const shared_ptr<Material>& material = shared_ptr<Material>());

    explicit Ellipse(double rx, double ry, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom);

    explicit Ellipse(const Ellipse& src) : GeometryObjectLeaf(src), radius0(src.radius0), radius1(src.radius1) {}

    shared_ptr<GeometryObject> shallowCopy() const override { return make_shared<Ellipse>(*this); }

    Box getBoundingBox() const override;

    bool contains(const DVec& p) const override;

    void addPointsAlongToSet(std::set<double>& points,
                             Primitive<3>::Direction direction,
                             unsigned max_steps,
                             double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

    /**
     * Get radii of this ellipse.
     * \return radii of this ellipse in form of pair (transverse, vertical)
     */
    std::pair<double, double> getRadii() const { return std::make_pair(radius0, radius1); }

    /**
     * Set radius and inform observers about changes.
     * \param rx new transverse radius to set
     * \param ry new vertical radius to set
     */
    void setRadii(double rx, double ry) {
        this->radius0 = std::max(rx, 0.);
        this->radius1 = std::max(ry, 0.);
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set transverse radius and inform observers about changes.
     * \param new_radius new transverse radius to set
     */
    void setRadius0(double new_radius) {
        if (new_radius < 0.) new_radius = 0.;
        this->radius0 = new_radius;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set vertical radius and inform observers about changes.
     * \param new_radius new vertical radius to set
     */
    void setRadius1(double new_radius) {
        if (new_radius < 0.) new_radius = 0.;
        this->radius1 = new_radius;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }
};

}  // namespace plask

#endif  // PLASK__GEOMETRY_ELLIPSE_H
