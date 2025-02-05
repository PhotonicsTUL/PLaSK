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
#ifndef PLASK__GEOMETRY_ELLIPTIC_CYLINDER_H
#define PLASK__GEOMETRY_ELLIPTIC_CYLINDER_H

#include "leaf.hpp"

namespace plask {

/**
 * EllipticCylinder with given height and base radius.
 *
 * Center of elliptic cylinders' base lies in point (0.0, 0.0, 0.0)
 */
struct PLASK_API EllipticCylinder : public GeometryObjectLeaf<3> {
    typedef typename GeometryObjectLeaf<3>::DVec DVec;

  private:

    DVec T(const DVec& p) const {
        return DVec(cosa * p.c0 - sina * p.c1, sina * p.c0 + cosa * p.c1, p.c2);
    }

    DVec TVec(double x, double y, double z) const {
        return DVec(cosa * x - sina * y, sina * x + cosa * y, z);
    }

    DVec invT(const DVec& p) const {
        return DVec(cosa * p.c0 + sina * p.c1, -sina * p.c0 + cosa * p.c1, p.c2);
    }

  public:
    double radius0;  ///< Longitudinal radius of this ellipse
    double radius1;  ///< Transverse radius of this ellipse
    double sina;     ///< sin of angle of rotation of the ellipse
    double cosa;     ///< cos of angle of rotation of the ellipse
    double height;   ///< Height of the cylinders

    static const char* NAME;

    std::string getTypeName() const override { return NAME; }

    EllipticCylinder(double radius0,
                     double radius1,
                     double angle,
                     double height,
                     const shared_ptr<Material>& material = shared_ptr<Material>());

    EllipticCylinder(double radius0,
                     double radius1,
                     double angle,
                     double height,
                     const shared_ptr<MaterialsDB::MixedCompositionFactory>& materialTopBottom);

    EllipticCylinder(double radius0, double radius1, double height, const shared_ptr<Material>& material = shared_ptr<Material>());

    EllipticCylinder(double radius0,
                     double radius1,
                     double height,
                     const shared_ptr<MaterialsDB::MixedCompositionFactory>& materialTopBottom);

    EllipticCylinder(const EllipticCylinder& src);

    Box getBoundingBox() const override;

    bool contains(const DVec& p) const override;

    // virtual bool intersects(const Box& area) const;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;

    shared_ptr<GeometryObject> shallowCopy() const override { return make_shared<EllipticCylinder>(*this); }

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

    /**
     * Get ellipse angle
     * \return angle of the ellipse
     */
    double getAngle() const { return atan2(sina, cosa); }

    /**
     * Set angle and inform observers about changes.
     * \param new_angle new angle to set
     */
    void setAngle(double new_angle) {
        this->sina = sin(new_angle);
        this->cosa = cos(new_angle);
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
     * Set radii, angle and height and inform observers about changes.
     * \param radius0 new radius0 to set
     * \param radius1 new radius1 to set
     * \param angle new angle to set
     * \param height new height to set
     */
    void resize(double radius0, double radius1, double angle, double height) {
        this->radius0 = radius0;
        this->radius1 = radius1;
        this->sina = sin(angle);
        this->cosa = cos(angle);
        this->height = height;
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Set radii and height and inform observers about changes.
     * \param radius0 new radius0 to set
     * \param radius1 new radius1 to set
     * \param angle new angle to set
     * \param height new height to set
     */
    void resize(double radius0, double radius1, double height) {
        this->radius0 = radius0;
        this->radius1 = radius1;
        this->sina = 0.0;
        this->cosa = 1.0;
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

#endif  // PLASK__GEOMETRY_ELLIPTIC_CYLINDER_H
