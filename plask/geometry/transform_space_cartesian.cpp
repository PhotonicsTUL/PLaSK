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
#include "transform_space_cartesian.hpp"

#include <algorithm>
#include "../manager.hpp"
#include "reader.hpp"

#define PLASK_EXTRUSION_NAME "extrusion"

namespace plask {

const char* Extrusion::NAME = PLASK_EXTRUSION_NAME;

std::string Extrusion::getTypeName() const { return NAME; }

void Extrusion::setLength(double new_length) {
    if (length == new_length) return;
    length = new_length;
    fireChanged(Event::EVENT_RESIZE);
}

bool Extrusion::contains(const DVec& p) const {
    return (this->hasChild() && canBeInside(p)) && this->_child->contains(childVec(p));
}

/*bool Extrusion::intersects(const Box& area) const {
    return canIntersect(area) && getChild()->intersects(childBox(area));
}*/

shared_ptr<Material> Extrusion::getMaterial(const DVec& p) const {
    return (this->hasChild() && canBeInside(p)) ? this->_child->getMaterial(childVec(p)) : shared_ptr<Material>();
}

Extrusion::Box Extrusion::fromChildCoords(const Extrusion::ChildType::Box& child_bbox) const {
    return parentBox(child_bbox);
}

/*std::vector< plask::shared_ptr< const plask::GeometryObject > > Extrusion::getLeafs() const {
    return getChild()->getLeafs();
}*/

shared_ptr<GeometryObject> Extrusion::shallowCopy() const {
    return shared_ptr<GeometryObjectTransform<3, Extrusion::ChildType>>(new Extrusion(this->_child, length));
}

GeometryObject::Subtree Extrusion::getPathsAt(const DVec& point, bool all) const {
    if (this->hasChild() && canBeInside(point))
        return GeometryObject::Subtree::extendIfNotEmpty(this, getChild()->getPathsAt(childVec(point), all));
    else
        return GeometryObject::Subtree();
}

void Extrusion::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    BaseClass::writeXMLAttr(dest_xml_object, axes);
    dest_xml_object.attr("length", length);
}

void Extrusion::getPositionsToVec(const GeometryObject::Predicate& predicate,
                                  std::vector<GeometryObjectTransformSpace::DVec>& dest,
                                  const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<3>::ZERO_VEC);
        return;
    }
    if (!this->hasChild()) return;
    auto child_pos_vec = this->_child->getPositions(predicate, path);
    for (const auto& v : child_pos_vec) dest.push_back(parentVec(v, std::numeric_limits<double>::quiet_NaN()));
}

// void Extrusion::extractToVec(const GeometryObject::Predicate &predicate, std::vector< shared_ptr<const
// GeometryObjectD<3> > >&dest, const PathHints *path) const {
//     if (predicate(*this)) {
//         dest.push_back(static_pointer_cast< const GeometryObjectD<3> >(this->shared_from_this()));
//         return;
//     }
//     std::vector< shared_ptr<const GeometryObjectD<2> > > child_res = getChild()->extract(predicate, path);
//     for (shared_ptr<const GeometryObjectD<2>>& c: child_res)
//         dest.emplace_back(new Extrusion(const_pointer_cast<GeometryObjectD<2>>(c), this->length));
// }

void Extrusion::addPointsAlongToSet(std::set<double>& points,
                               Primitive<3>::Direction direction,
                               unsigned max_steps,
                               double min_step_size) const {
    if (!this->hasChild()) return;
    if (direction == Primitive<3>::DIRECTION_LONG) {
        points.insert(0);
        points.insert(length);
    } else {
        if (this->max_steps) max_steps = this->max_steps;
        if (this->min_step_size) min_step_size = this->min_step_size;
        _child->addPointsAlongToSet(points, direction, max_steps, min_step_size);
    }
}

void Extrusion::addLineSegmentsToSet(std::set<typename GeometryObjectD<3>::LineSegment>& segments,
                                     unsigned max_steps,
                                     double min_step_size) const {
    if (!this->hasChild()) return;
    typedef typename GeometryObjectD<3>::LineSegment Segment3;
    typedef typename GeometryObjectD<2>::LineSegment Segment2;
    if (this->max_steps) max_steps = this->max_steps;
    if (this->min_step_size) min_step_size = this->min_step_size;
    std::set<Segment2> child_segments;
    _child->addLineSegmentsToSet(child_segments, max_steps, min_step_size);
    for (const auto& s: child_segments) {
        segments.insert(Segment3(DVec(0., s.p0().c0, s.p0().c1), DVec(0., s.p1().c0, s.p1().c1)));
        segments.insert(Segment3(DVec(0., s.p0().c0, s.p0().c1), DVec(length, s.p0().c0, s.p0().c1)));
        segments.insert(Segment3(DVec(0., s.p1().c0, s.p1().c1), DVec(length, s.p1().c0, s.p1().c1)));
        segments.insert(Segment3(DVec(length, s.p0().c0, s.p0().c1), DVec(length, s.p1().c0, s.p1().c1)));
    }
}

shared_ptr<GeometryObject> read_cartesianExtend(GeometryReader& reader) {
    double length = reader.source.requireAttribute<double>("length");
    GeometryReader::SetExpectedSuffix suffixSetter(reader, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
    return plask::make_shared<Extrusion>(
        reader.readExactlyOneChild<typename Extrusion::ChildType>(!reader.manager.draft), length);
}

static GeometryReader::RegisterObjectReader cartesianExtend2D_reader(PLASK_EXTRUSION_NAME, read_cartesianExtend);

}  // namespace plask
