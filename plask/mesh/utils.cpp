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
#include "utils.hpp"

namespace plask {

void plask::SameMeshChecker::setMesh(const plask::Mesh *mesh) {
    connection_with_mesh.disconnect();
    this->mesh = mesh;
    if (this->mesh)
        connection_with_mesh = const_cast<Mesh*>(this->mesh)->changed.connect(boost::bind(&SameMeshChecker::onMeshChanged, this, _1), boost::signals2::at_front);
}

}   // namespace plask
