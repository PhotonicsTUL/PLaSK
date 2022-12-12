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
#ifndef PLASK__GEOMETRY_CYLINDER_H
#define PLASK__GEOMETRY_CYLINDER_H

#include "leaf.hpp"

namespace plask {

/**
 * Cylinder with given height and radius of base.
 *
 * Center of cylinders' base lies in point (0.0, 0.0, 0.0)
 */
struct PLASK_API Cylinder : public GeometryObjectLeaf<3> {
    double radius, height;

    static const char* NAME;

    std::string getTypeName() const override { return NAME; }

    Cylinder(double radius, double height, const shared_ptr<Material>& material = shared_ptr<Material>());

    Cylinder(double radius, double height, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom);

    Cylinder(const Cylinder& src);

    Box getBoundingBox() const override;

    bool contains(const DVec& p) const override;

    // virtual bool intersects(const Box& area) const;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

    shared_ptr<GeometryObject> shallowCopy() const override { return make_shared<Cylinder>(*this); }

    /**
     * Set radius and inform observers about changes.
     * @param new_radius new radius to set
     */
    void setRadius(double new_radius) {
        if (new_radius < 0.) new_radius = 0.;
        this->radius = new_radius;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set height and inform observers about changes.
     * @param new_height new height to set
     */
    void setHeight(double new_height) {
        if (new_height < 0.) new_height = 0.;
        this->height = new_height;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set radius and height and inform observers about changes.
     * @param radius new radius to set
     * @param height new height to set
     */
    void resize(double radius, double height) {
        this->radius = radius;
        this->height = height;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    void addPointsAlongToSet(std::set<double>& points,
                             Primitive<3>::Direction direction,
                             unsigned max_steps,
                             double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;
};

}  // namespace plask

#endif  // PLASK__GEOMETRY_CYLINDER_H
