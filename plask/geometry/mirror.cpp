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
#include "mirror.hpp"
#include "../manager.hpp"
#include "reader.hpp"

#define PLASK_FLIP2D_NAME ("flip" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D)
#define PLASK_FLIP3D_NAME ("flip" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D)

#define PLASK_MIRROR2D_NAME ("mirror" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D)
#define PLASK_MIRROR3D_NAME ("mirror" PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D)

namespace plask {

template <int dim> const char* Flip<dim>::NAME = dim == 2 ? PLASK_FLIP2D_NAME : PLASK_FLIP3D_NAME;

template <int dim> std::string Flip<dim>::getTypeName() const { return NAME; }

template <int dim> shared_ptr<Material> Flip<dim>::getMaterial(const typename Flip<dim>::DVec& p) const {
    return this->hasChild() ? this->_child->getMaterial(flipped(p)) : shared_ptr<Material>();
}

template <int dim> bool Flip<dim>::contains(const typename Flip<dim>::DVec& p) const {
    return this->hasChild() && this->_child->contains(flipped(p));
}

template <int dim>
GeometryObject::Subtree Flip<dim>::getPathsAt(const typename Flip<dim>::DVec& point, bool all) const {
    if (!this->hasChild()) return GeometryObject::Subtree();
    return GeometryObject::Subtree::extendIfNotEmpty(this, this->_child->getPathsAt(flipped(point), all));
}

template <int dim>
typename Flip<dim>::Box Flip<dim>::fromChildCoords(const typename Flip<dim>::ChildType::Box& child_bbox) const {
    return flipped(child_bbox);
}

template <int dim>
void Flip<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate,
                                  std::vector<DVec>& dest,
                                  const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    if (!this->hasChild()) return;
    std::size_t s = dest.size();
    this->_child->getPositionsToVec(predicate, dest, path);
    for (; s < dest.size(); ++s)
        dest[s][flipDir] =
            std::numeric_limits<double>::quiet_NaN();  // we can't get proper position in flipDir direction
}

template <int dim> shared_ptr<GeometryObject> Flip<dim>::shallowCopy() const { return copyShallow(); }

template <int dim>
void Flip<dim>::addPointsAlongToSet(std::set<double>& points,
                                    Primitive<3>::Direction direction,
                                    unsigned max_steps,
                                    double min_step_size) const {
    if (this->_child) {
        if (int(direction) == int(flipDir) + 3 - dim) {
            std::set<double> child_points;
            this->_child->addPointsAlongToSet(child_points, direction, max_steps, min_step_size);
            for (double p : child_points) points.insert(-p);
        } else {
            this->_child->addPointsAlongToSet(points, direction, max_steps, min_step_size);
        }
    }
}

template <int dim>
void Flip<dim>::addLineSegmentsToSet(std::set<typename GeometryObjectD<dim>::LineSegment>& segments,
                                     unsigned max_steps,
                                     double min_step_size) const {
    if (this->_child) {
        std::set<typename GeometryObjectD<dim>::LineSegment> child_segments;
        this->_child->addLineSegmentsToSet(child_segments, this->max_steps ? this->max_steps : max_steps,
                                           this->min_step_size ? this->min_step_size : min_step_size);
        for (const auto& p : child_segments)
            segments.insert(typename GeometryObjectD<dim>::LineSegment(flipped(p[0]), flipped(p[1])));
    }
}

template <int dim> void Flip<dim>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    BaseClass::writeXMLAttr(dest_xml_object, axes);
    dest_xml_object.attr("axis", axes[direction3D(flipDir)]);
}

template <int dim> const char* Mirror<dim>::NAME = dim == 2 ? PLASK_MIRROR2D_NAME : PLASK_MIRROR3D_NAME;

template <int dim> std::string Mirror<dim>::getTypeName() const { return NAME; }

template <int dim> typename Mirror<dim>::Box Mirror<dim>::getBoundingBox() const {
    return this->hasChild() ? extended(this->_child->getBoundingBox())
                            : Box(Primitive<dim>::ZERO_VEC, Primitive<dim>::ZERO_VEC);
}

template <int dim> typename Mirror<dim>::Box Mirror<dim>::getRealBoundingBox() const {
    return this->hasChild() ? this->_child->getBoundingBox() : Box(Primitive<dim>::ZERO_VEC, Primitive<dim>::ZERO_VEC);
}

template <int dim> shared_ptr<Material> Mirror<dim>::getMaterial(const typename Mirror<dim>::DVec& p) const {
    return this->hasChild() ? this->_child->getMaterial(flippedIfNeg(p)) : shared_ptr<Material>();
}

template <int dim> bool Mirror<dim>::contains(const typename Mirror<dim>::DVec& p) const {
    return this->hasChild() && this->_child->contains(flippedIfNeg(p));
}

template <int dim>
typename Mirror<dim>::Box Mirror<dim>::fromChildCoords(const typename Mirror<dim>::ChildType::Box& child_bbox) const {
    return child_bbox.extension(child_bbox.flipped(flipDir));
}

template <int dim>
void Mirror<dim>::getBoundingBoxesToVec(const GeometryObject::Predicate& predicate,
                                        std::vector<Box>& dest,
                                        const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(getBoundingBox());
        return;
    }
    if (!this->hasChild()) return;
    std::size_t old_size = dest.size();
    this->_child->getBoundingBoxesToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (std::size_t i = old_size; i < new_size; ++i) dest.push_back(dest[i].flipped(flipDir));
}

template <int dim>
void Mirror<dim>::getObjectsToVec(const GeometryObject::Predicate& predicate,
                                  std::vector<shared_ptr<const GeometryObject>>& dest,
                                  const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(this->shared_from_this());
        return;
    }
    if (!this->hasChild()) return;
    std::size_t old_size = dest.size();
    this->_child->getObjectsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (std::size_t i = old_size; i < new_size; ++i) dest.push_back(dest[i]);
}

template <int dim>
void Mirror<dim>::getPositionsToVec(const GeometryObject::Predicate& predicate,
                                    std::vector<DVec>& dest,
                                    const PathHints* path) const {
    if (predicate(*this)) {
        dest.push_back(Primitive<dim>::ZERO_VEC);
        return;
    }
    if (!this->hasChild()) return;
    std::size_t old_size = dest.size();
    this->_child->getPositionsToVec(predicate, dest, path);
    std::size_t new_size = dest.size();
    for (std::size_t i = old_size; i < new_size; ++i) {
        dest.push_back(dest[i]);
        dest.back()[flipDir] =
            std::numeric_limits<double>::quiet_NaN();  // we can't get proper position in flipDir direction
    }
}

template <int dim>
GeometryObject::Subtree Mirror<dim>::getPathsTo(const GeometryObject& el, const PathHints* path) const {
    GeometryObject::Subtree result = GeometryObjectTransform<dim>::getPathsTo(el, path);
    if (!result.empty() && !result.children.empty())  // result.children[0] == getChild()
        result.children.push_back(
            GeometryObject::Subtree(plask::make_shared<Flip<dim>>(flipDir, getChild()), result.children[0].children));
    return result;
}

template <int dim>
GeometryObject::Subtree Mirror<dim>::getPathsAt(const typename Mirror<dim>::DVec& point, bool all) const {
    if (!this->hasChild()) GeometryObject::Subtree();
    return GeometryObject::Subtree::extendIfNotEmpty(this, this->_child->getPathsAt(flippedIfNeg(point), all));
}

template <int dim> std::size_t Mirror<dim>::getChildrenCount() const { return this->hasChild() ? 2 : 0; }

template <int dim> shared_ptr<GeometryObject> Mirror<dim>::getChildNo(std::size_t child_no) const {
    if (child_no >= getChildrenCount())
        throw OutOfBoundsException("getChildNo", "child_no", child_no, 0, getChildrenCount() - 1);
    // child_no is 0 or 1 now, and hasChild() is true
    if (child_no == 0) return this->_child;
    else
        return plask::make_shared<Flip<dim>>(flipDir, this->_child);
}

template <int dim> std::size_t Mirror<dim>::getRealChildrenCount() const {
    return GeometryObjectTransform<dim>::getChildrenCount();
}

template <int dim> shared_ptr<GeometryObject> Mirror<dim>::getRealChildNo(std::size_t child_no) const {
    return GeometryObjectTransform<dim>::getChildNo(child_no);
}

template <int dim> shared_ptr<GeometryObject> Mirror<dim>::shallowCopy() const { return copyShallow(); }

template <int dim>
void Mirror<dim>::addPointsAlongToSet(std::set<double>& points,
                                      Primitive<3>::Direction direction,
                                      unsigned max_steps,
                                      double min_step_size) const {
    if (this->_child) {
        if (this->max_steps) max_steps = this->max_steps;
        if (this->min_step_size) min_step_size = this->min_step_size;
        if (int(direction) == int(flipDir) + 3 - dim) {
            std::set<double> child_points;
            this->_child->addPointsAlongToSet(child_points, direction, max_steps, min_step_size);
            for (double p : child_points) points.insert(-p);
            for (double p : child_points) points.insert(p);
        } else {
            this->_child->addPointsAlongToSet(points, direction, max_steps, min_step_size);
        }
    }
}

template <int dim>
void Mirror<dim>::addLineSegmentsToSet(std::set<typename GeometryObjectD<dim>::LineSegment>& segments,
                                       unsigned max_steps,
                                       double min_step_size) const {
    if (this->_child) {
        std::set<typename GeometryObjectD<dim>::LineSegment> child_segments;
        this->_child->addLineSegmentsToSet(child_segments, this->max_steps ? this->max_steps : max_steps,
                                           this->min_step_size ? this->min_step_size : min_step_size);
        for (const auto& p : child_segments) {
            segments.insert(typename GeometryObjectD<dim>::LineSegment(flipped(p[0]), flipped(p[1])));
            segments.insert(p);
        }
    }
}

template <int dim> void Mirror<dim>::writeXMLAttr(XMLWriter::Element& dest_xml_object, const AxisNames& axes) const {
    BaseClass::writeXMLAttr(dest_xml_object, axes);
    dest_xml_object.attr("axis", axes[direction3D(flipDir)]);
}

//--------- XML reading: Flip and Mirror ----------------

template <typename GeometryType> shared_ptr<GeometryObject> read_flip_like(GeometryReader& reader) {
    GeometryReader::SetExpectedSuffix suffixSetter(
        reader, GeometryType::DIM == 2 ? PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D : PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
    auto flipDir = reader.getAxisNames().get<GeometryType::DIM>(reader.source.requireAttribute("axis"));
    return plask::make_shared<GeometryType>(
        flipDir, reader.readExactlyOneChild<typename GeometryType::ChildType>(!reader.manager.draft));
}

static GeometryReader::RegisterObjectReader flip2D_reader(PLASK_FLIP2D_NAME, read_flip_like<Flip<2>>);
static GeometryReader::RegisterObjectReader flip3D_reader(PLASK_FLIP3D_NAME, read_flip_like<Flip<3>>);
static GeometryReader::RegisterObjectReader mirror2D_reader(PLASK_MIRROR2D_NAME, read_flip_like<Mirror<2>>);
static GeometryReader::RegisterObjectReader mirror3D_reader(PLASK_MIRROR3D_NAME, read_flip_like<Mirror<3>>);

template struct PLASK_API Flip<2>;
template struct PLASK_API Flip<3>;

template struct PLASK_API Mirror<2>;
template struct PLASK_API Mirror<3>;

}  // namespace plask
