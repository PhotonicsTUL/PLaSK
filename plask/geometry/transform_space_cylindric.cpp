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
#include "transform_space_cylindric.hpp"
#include "../manager.hpp"
#include "reader.hpp"

#define PLASK_REVOLUTION_NAME "revolution"

namespace plask {

const char* Revolution::NAME = PLASK_REVOLUTION_NAME;

std::string Revolution::getTypeName() const { return NAME; }

bool Revolution::contains(const GeometryObjectD<3>::DVec& p) const {
    return this->hasChild() && this->_child->contains(childVec(p));
}

/*bool Revolution::intersects(const Box& area) const {
    return getChild()->intersects(childBox(area));
}*/

shared_ptr<Material> Revolution::getMaterial(const DVec& p) const {
    return this->hasChild() ? this->_child->getMaterial(childVec(p)) : shared_ptr<Material>();
}

Revolution::Box Revolution::fromChildCoords(const Revolution::ChildType::Box& child_bbox) const {
    return parentBox(child_bbox);
}

shared_ptr<GeometryObject> Revolution::shallowCopy() const { return plask::make_shared<Revolution>(this->_child); }

GeometryObject::Subtree Revolution::getPathsAt(const DVec& point, bool all) const {
    if (!this->hasChild()) return GeometryObject::Subtree();
    return GeometryObject::Subtree::extendIfNotEmpty(this, this->_child->getPathsAt(childVec(point), all));
}

void Revolution::getPositionsToVec(const GeometryObject::Predicate& predicate,
                                   std::vector<GeometryObjectTransformSpace::DVec>& dest,
                                   const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<3>::ZERO_VEC);
        return;
    }
    if (!this->hasChild()) return;
    auto child_pos_vec = this->_child->getPositions(predicate, path);
    for (const auto& v : child_pos_vec)
        dest.emplace_back(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
                          v.vert()  // only vert component is well defined
        );
}

bool Revolution::childIsClipped() const {
    return this->hasChild() && (this->_child->getBoundingBox().lower.tran() < 0);
}

// void Revolution::extractToVec(const GeometryObject::Predicate &predicate, std::vector< shared_ptr<const
// GeometryObjectD<3> > >&dest, const PathHints *path) const {
//     if (predicate(*this)) {
//         dest.push_back(static_pointer_cast< const GeometryObjectD<3> >(this->shared_from_this()));
//         return;
//     }
//     std::vector< shared_ptr<const GeometryObjectD<2> > > child_res = getChild()->extract(predicate, path);
//     for (shared_ptr<const GeometryObjectD<2>>& c: child_res)
//         dest.emplace_back(new Revolution(const_pointer_cast<GeometryObjectD<2>>(c)));
// }

/*Box2D Revolution::childBox(const plask::Box3D& r) {
    Box2D result(childVec(r.lower), childVec(r.upper));
    result.fix();
    return result;
}*/ //TODO bugy

Box3D Revolution::parentBox(const ChildBox& r) {
    double tran = std::max(r.upper.tran(), 0.0);
    return Box3D(vec(-tran, -tran, r.lower.vert()), vec(tran, tran, r.upper.vert()));
}

void Revolution::addPointsAlongToSet(std::set<double>& points,
                                     Primitive<3>::Direction direction,
                                     unsigned max_steps,
                                     double min_step_size) const {
    if (!this->hasChild()) return;
    if (this->max_steps) max_steps = this->max_steps;
    if (this->min_step_size) min_step_size = this->min_step_size;
    if (direction == Primitive<3>::DIRECTION_VERT) {
        this->_child->addPointsAlongToSet(points, Primitive<3>::DIRECTION_VERT, max_steps, min_step_size);
    } else {
        std::set<double> child_points;
        this->_child->addPointsAlongToSet(child_points, Primitive<3>::DIRECTION_TRAN, max_steps, min_step_size);
        if (child_points.size() == 0) return;
        // Finer separation
        std::vector<double> pts;
        pts.reserve(child_points.size());
        pts.insert(pts.end(), child_points.begin(), child_points.end());
        double rr = pts[pts.size() - 1] - pts[0];
        for (size_t i = 1; i < pts.size(); ++i) {
            double r = pts[i - 1];
            double dr = pts[i] - r;
            unsigned maxsteps = rev_max_steps * (dr / rr);
            unsigned steps = min(unsigned(dr / rev_min_step_size), maxsteps);
            double step = dr / steps;
            for (unsigned j = 0; j < steps; ++j) {
                points.insert(-r - j * step);
                points.insert(r + j * step);
            }
        }
        points.insert(-pts[pts.size() - 1]);
        points.insert(pts[pts.size() - 1]);
    }
}

void Revolution::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                      unsigned max_steps,
                                      double min_step_size) const {
    if (!this->hasChild()) return;
    typedef typename GeometryObjectD<3>::LineSegment Segment;
    if (this->max_steps) max_steps = this->max_steps;
    if (this->min_step_size) min_step_size = this->min_step_size;
    std::set<typename GeometryObjectD<2>::LineSegment> segments2;
    this->_child->addLineSegmentsToSet(segments2, max_steps, min_step_size);
    double radius = max(abs(this->_child->getBoundingBox().left()), abs(this->_child->getBoundingBox().right()));
    unsigned steps = min(unsigned(M_PI * radius / min_step_size), max_steps);
    double dphi = M_PI / steps;
    double cos0 = 1., sin0 = 0;
    for (unsigned i = 1; i <= (steps + 1) / 2; ++i) {
        double phi = dphi * i;
        double cos1 = cos(phi), sin1 = sin(phi);
        for (auto seg : segments2) {
            double x[2], y[2], z[2];
            for (int j = 0; j < 2; ++i) {
                x[j] = seg[j].c0 * cos1;
                y[j] = seg[j].c0 * sin1;
                z[j] = seg[j].c1;
                double x0 = seg[j].c0 * cos0, y0 = seg[j].c0 * cos0;
                segments.insert(Segment(DVec(-x0, -y0, z[j]), DVec(-x[j], -y[j], z[j])));
                segments.insert(Segment(DVec(x0, -y0, z[j]), DVec(x[j], -y[j], z[j])));
                segments.insert(Segment(DVec(-x0, y0, z[j]), DVec(-x[j], y[j], z[j])));
                segments.insert(Segment(DVec(x0, y0, z[j]), DVec(x[j], y[j], z[j])));
            }
            segments.insert(Segment(DVec(-x[0], -y[0], z[0]), DVec(-x[1], -y[1], z[1])));
            segments.insert(Segment(DVec(x[0], -y[0], z[0]), DVec(x[1], -y[1], z[1])));
            segments.insert(Segment(DVec(-x[0], y[0], z[0]), DVec(-x[1], y[1], z[1])));
            segments.insert(Segment(DVec(x[0], y[0], z[0]), DVec(x[1], y[1], z[1])));
        }
        cos0 = cos1;
        sin0 = sin1;
    }
}

shared_ptr<GeometryObject> read_revolution(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    bool auto_clip = reader.source.getAttribute("auto-clip", false);
    auto rev_max_steps = reader.source.getAttribute<unsigned>("rev-steps-num");
    auto rev_min_step_size = reader.source.getAttribute<double>("rev-steps-dist");
    auto revolution = plask::make_shared<Revolution>(
        reader.readExactlyOneChild<typename Revolution::ChildType>(), auto_clip);
    if (rev_max_steps) revolution->rev_max_steps = *rev_max_steps;
    if (rev_min_step_size) revolution->rev_min_step_size = *rev_min_step_size;
    return revolution;
    /*if (res->childIsClipped()) {
        writelog(LOG_WARNING, "Child of <revolution>, read from XPL line {0}, is implicitly clipped (to non-negative
    tran. coordinates).", line_nr);
    }*/
}

static GeometryReader::RegisterObjectReader revolution_reader(PLASK_REVOLUTION_NAME, read_revolution);

}  // namespace plask
