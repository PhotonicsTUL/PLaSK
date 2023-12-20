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
#ifndef PLASK__TRANSFORM_SPACE_CYLINDRIC_H
#define PLASK__TRANSFORM_SPACE_CYLINDRIC_H

#include "transform.hpp"

namespace plask {

/**
 * Represent 3D geometry object which is an effect of revolving a 2D object (child) around the up axis.
 *
 * Child should have getBoundingBox().lower.tran() >= 0. When it doesn't have, it is implicitly clipped.
 * @ingroup GEOMETRY_OBJ
 */
struct PLASK_API Revolution : public GeometryObjectTransformSpace<3, 2> {
    unsigned rev_max_steps;
    double rev_min_step_size;

    /**
     * @param child object to revolve
     * @param auto_clip if false child must have getBoundingBox().lower.tran() >= 0, if true it will be clipped
     */
    Revolution(shared_ptr<ChildType> child = shared_ptr<ChildType>(), bool auto_clip = false)
        : GeometryObjectTransformSpace<3, 2>(child),
          rev_max_steps(PLASK_GEOMETRY_MAX_STEPS),
          rev_min_step_size(PLASK_GEOMETRY_MIN_STEP_SIZE) {
        if (!auto_clip && childIsClipped())
            throw Exception(
                "Child of Revolution must have bounding box with positive tran. coordinates (when auto clipping is "
                "off).");
    }

    static const char* NAME;

    /// Set max_steps for revolution
    void setRevMaxSteps(unsigned long value) {
        rev_max_steps = value;
        fireChanged(GeometryObject::Event::EVENT_STEPS);
    }

    /// Set min_step_size for revolution
    void setRevMinStepSize(double value) {
        rev_min_step_size = value;
        fireChanged(GeometryObject::Event::EVENT_STEPS);
    }

    std::string getTypeName() const override;

    bool contains(const DVec& p) const override;

    // TODO good but unused
    // virtual bool intersects(const Box& area) const;

    shared_ptr<Material> getMaterial(const DVec& p) const override;

    Box fromChildCoords(const typename ChildType::Box& child_bbox) const override;

    shared_ptr<GeometryObject> shallowCopy() const override;

    using GeometryObjectTransformSpace<3, 2>::getPathsTo;

    GeometryObject::Subtree getPathsAt(const DVec& point, bool all = false) const override;

    void getPositionsToVec(const GeometryObject::Predicate& predicate,
                           std::vector<DVec>& dest,
                           const PathHints* path = 0) const override;

    // virtual void extractToVec(const GeometryObject::Predicate& predicate, std::vector< shared_ptr<const
    // GeometryObjectD<dim> > >& dest, const PathHints* = 0) const;

    /**
     * Convert vector @p v to space of child.
     * @param r vector in parent (this) space, i.e. in space where (0, 0, 0) is at center of base of cylinder
     * @return vector in child space
     */
    static Vec<2, double> childVec(const Vec<3, double>& v) { return rotateToLonTranAbs(v); }

    /*
     * Convert rectangle @p r to space of child.
     * @param r cuboid in parent (this) space
     * @return rectangle in child space
     */
    // static ChildBox childBox(const Box& r);

    /**
     * Check if child can be clipped.
     * @return @c true if child has getBoundingBox().lower.tran() < 0
     */
    bool childIsClipped() const;

  private:
    /**
     * Convert rectangle @p r to space of parent (this).
     * @param r rectangle in child space
     * @return cuboid in parent (this) space
     */
    static Box parentBox(const ChildBox& r);

    void addPointsAlongToSet(std::set<double>& points,
                        Primitive<3>::Direction direction,
                        unsigned max_steps,
                        double min_step_size) const override;

    void addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                              unsigned max_steps,
                              double min_step_size) const override;
};

}  // namespace plask

#endif  // PLASK__TRANSFORM_SPACE_CYLINDRIC_H
