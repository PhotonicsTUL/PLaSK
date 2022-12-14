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
#include "basic.hpp"

namespace plask {

template <int DIM>
std::size_t OnePointMesh<DIM>::size() const {
    return 1;
}

template <int DIM>
typename OnePointMesh<DIM>::DVec OnePointMesh<DIM>::at(std::size_t) const {
    return point;
}

template struct PLASK_API OnePointMesh<2>;
template struct PLASK_API OnePointMesh<3>;

template <>
void OnePointMesh<3>::writeXML(XMLElement& object) const {
    object.attr("type", "point3d"); // this is required attribute for the provided object
    object.addTag("point")
           .attr("c0", point.c0)
           .attr("c1", point.c1)
           .attr("c2", point.c2);
}

template <>
void OnePointMesh<2>::writeXML(XMLElement &object) const {
    object.attr("type", "point2d"); // this is required attribute for the provided object
    object.addTag("point")
           .attr("c0", point.c0)
           .attr("c1", point.c1);
}

static shared_ptr<Mesh> readOnePoint3DMesh(XMLReader& reader) {
    reader.requireTag("point");
    double c0 = reader.requireAttribute<double>("c0");
    double c1 = reader.requireAttribute<double>("c1");
    double c2 = reader.requireAttribute<double>("c2");
    reader.requireTagEnd();   // this is necessary to make sure the tag <point> is closed
    // Now create the mesh into a shared pointer and return it:
    return plask::make_shared<OnePointMesh<3>>(vec(c0, c1, c2));
}

static RegisterMeshReader onepoint3dmesh_reader("point3d", &readOnePoint3DMesh);

static shared_ptr<Mesh> readOnePoint2DMesh(XMLReader& reader) {
    reader.requireTag("point");
    double c0 = reader.requireAttribute<double>("c0");
    double c1 = reader.requireAttribute<double>("c1");
    reader.requireTagEnd();   // this is necessary to make sure the tag <point> is closed
    // Now create the mesh into a shared pointer and return it:
    return plask::make_shared<OnePointMesh<2>>(vec(c0, c1));
}

static RegisterMeshReader onepoint2dmesh_reader("point2d", &readOnePoint2DMesh);

template <int DIM>
typename TranslatedMesh<DIM>::DVec TranslatedMesh<DIM>::at(std::size_t index) const {
    return sourceMesh->at(index) + translation;
}

template <int DIM>
std::size_t TranslatedMesh<DIM>::size() const {
    return sourceMesh->size();
}

template struct PLASK_API TranslatedMesh<2>;
template struct PLASK_API TranslatedMesh<3>;



}   // namespace plask
