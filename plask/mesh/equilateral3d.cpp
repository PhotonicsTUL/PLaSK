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
#include "equilateral3d.hpp"
#include "equilateral3d.hpp"

namespace plask {

EquilateralMesh3D::EquilateralMesh3D(IterationOrder iterationOrder, Vec<3> vec0, Vec<3> vec1, Vec<3> vec2):
    RectilinearMesh3D(iterationOrder), trans{vec0.c0, vec1.c0, vec2.c0, vec0.c1, vec1.c1, vec2.c1, vec0.c2, vec1.c2, vec2.c2} {
        findInverse();
    }

EquilateralMesh3D::EquilateralMesh3D(shared_ptr<MeshAxis> mesh0, shared_ptr<MeshAxis> mesh1, shared_ptr<MeshAxis> mesh2, IterationOrder iterationOrder,
                                     Vec<3> vec0, Vec<3> vec1, Vec<3> vec2):
    RectilinearMesh3D(std::move(mesh0), std::move(mesh1), std::move(mesh2), iterationOrder),
    trans{vec0.c0, vec1.c0, vec2.c0, vec0.c1, vec1.c1, vec2.c1, vec0.c2, vec1.c2, vec2.c2} {
        findInverse();
    }

shared_ptr<EquilateralMesh3D::ElementMesh> EquilateralMesh3D::getElementMesh() const {
    return plask::make_shared<EquilateralMesh3D::ElementMesh>(this, axis[0]->getMidpointAxis(), axis[1]->getMidpointAxis(), axis[2]->getMidpointAxis(),
                                                              getIterationOrder(), getVec0(), getVec1(), getVec2());
}

} // namespace plask
