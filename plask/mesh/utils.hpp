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
#ifndef PLASK__MESH_UTILS_H
#define PLASK__MESH_UTILS_H

#include "mesh.hpp"

/** @file
This file contains some utils usefull with mesh classes.
*/

namespace plask {

/**
 * Object of this class alows for checking if a given mesh is the same mesh which has been used recently.
 */
struct PLASK_API SameMeshChecker {

private:

    /// Connection with mesh change callback.
    boost::signals2::connection connection_with_mesh;

    /// Mesh which was given recently.
    const Mesh* mesh;

    void setMesh(const Mesh* mesh);

    /**
     * Refresh bounding box cache. Called by mesh changed signal.
     */
    void onMeshChanged(const Mesh::Event&) {
        setMesh(nullptr);
    }

public:

    /**
     * Check if this operator has been recently called with given @p mesh and this @p mesh has not been changed since that time.
     * @param mesh mesh to check
     * @return @c true only if this operator has been recently called with given @p mesh and this @p mesh has not been changed since that time.
     */
    bool operator()(const Mesh* mesh) {
        if (this->mesh == mesh)
            return true;
        else {
            setMesh(mesh);
            return false;
        }
    }

    /**
     * Check if this operator has been recently called with given @p mesh and this @p mesh has not been changed since that time.
     * @param mesh mesh to check
     * @return @c true only if this operator has been recently called with given @p mesh and this @p mesh has not been changed since that time.
     */
    bool operator()(shared_ptr<const Mesh> mesh) {
        return this->operator()(mesh.get());
    }

    /*
     * Get mesh for which operator() has been recently called if it this @p mesh has not been changed.
     * @return mesh for which operator() has been recently called if it this @p mesh has not been changed.
     */
    /*const Mesh* getMesh() const {
        return mesh;
    }*/

    /**
     * Constructor.
     */
    SameMeshChecker(): mesh(nullptr) {}

};


}   // namespace plask

#endif // PLASK__MESH_UTILS_H

