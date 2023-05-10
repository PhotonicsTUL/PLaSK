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
#ifndef PLASK__GEOMETRY_CUBOID_H
#define PLASK__GEOMETRY_CUBOID_H

/** @file
This file contains a rotated cuboid (geometry object) class.
*/

#include "leaf.hpp"
#include "../math.hpp"


namespace plask {

/**
Represent a cuboid that can be rotated in a horizontal plane

@ingroup GEOMETRY_OBJ
*/
struct PLASK_API RotatedCuboid : public Block<3> {
    /// Vector of doubles type in space on this, vector in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<3>::DVec DVec;

    /// Rectangle type in space on this, rectangle in space with dim number of dimensions.
    typedef typename GeometryObjectLeaf<3>::Box Box;

    static const char* NAME;

  protected:
    /// Cosine of the rotation angle
    double c;

    /// Sine of the rotation angle
    double s;

  public:
    /**
     * Transform a local (rotated) point to global coordinates
     * \param c0,c1,c2 coordinates to transform
     * \return transformed vector
     */
    DVec trans(double c0, double c1, double c2 = 0.) const { return DVec(c * c0 - s * c1, s * c0 + c * c1, c2); }

    /**
     * Transform a local (rotated) point to global coordinates
     * \param vec coordinates vector to transform
     * \return transformed vector
     */
    DVec trans(const DVec& vec) const { return trans(vec.c0, vec.c1, vec.c2); }

    /**
     * Transform a point to local (rotated) coordinates
     * \param c0,c1,c2 coordinates to transform
     * \return transformed vector
     */
    DVec itrans(double c0, double c1, double c2 = 0.) const { return DVec(c * c0 + s * c1, -s * c0 + c * c1, c2); }

    /**
     * Transform a point to local (rotated) coordinates
     * \param vec coordinates vector to transform
     * \return transformed vector
     */
    DVec itrans(const DVec& vec) const { return itrans(vec.c0, vec.c1, vec.c2); }

    std::string getTypeName() const override;

    /**
     * Get rotation angle.
     * \return angle new angle to set [deg]
     */
    double getAngle() const { return 180. / M_PI * atan2(s, c); }

    /**
     * Set rotation angle and inform observers about changes.
     * \param angle new angle to set [deg]
     */
    void setAngle(double angle) {
        double rot = M_PI / 180. * angle;
        c = cos(rot);
        s = sin(rot);
        this->fireChanged(GeometryObject::Event::EVENT_RESIZE);
    }

    /**
     * Create cuboid.
     * \param size size/upper corner of block
     * \param angle rotation angle [deg]
     * \param material block material
     */
    explicit RotatedCuboid(const DVec& size = Primitive<3>::ZERO_VEC,
                           double angle = 0.,
                           const shared_ptr<Material>& material = shared_ptr<Material>())
        : Block<3>(size, material) {
        double rot = M_PI / 180. * angle;
        c = cos(rot);
        s = sin(rot);
    }

    explicit RotatedCuboid(double angle) {
        double rot = M_PI / 180. * angle;
        c = cos(rot);
        s = sin(rot);
    }

    explicit RotatedCuboid(const DVec& size, double angle, shared_ptr<MaterialsDB::MixedCompositionFactory> materialTopBottom)
        : Block<3>(size, materialTopBottom) {
        double rot = M_PI / 180. * angle;
        c = cos(rot);
        s = sin(rot);
    }

    explicit RotatedCuboid(const Block<3>& src) : Block<3>(src), c(1), s(0) {}

    explicit RotatedCuboid(const RotatedCuboid& src) : Block<3>(src), c(src.c), s(src.s) {}

    Box getBoundingBox() const override;

    bool contains(const DVec& p) const override;

    shared_ptr<GeometryObject> shallowCopy() const override { return make_shared<RotatedCuboid>(*this); }

    void addPointsAlongToSet(std::set<double>& points,
                             Primitive<3>::Direction direction,
                             unsigned max_steps,
                             double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;

    void writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const override;
};

shared_ptr<GeometryObject> read_cuboid(GeometryReader& reader);

}  // namespace plask

#endif
