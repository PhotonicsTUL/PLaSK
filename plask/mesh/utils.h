#ifndef PLASK__MESH_UTILS_H
#define PLASK__MESH_UTILS_H

#include "mesh.h"

/** @file
This file contains some utils usefull with mesh classes.
*/

namespace plask {

/**
 * Object of this class alows to check if mesh is the same mesh which has been used recently.
 */
struct SameMeshChecker {

private:

    /// Connection with mesh change callback.
    boost::signals2::connection connection_with_mesh;

    /// Mesh which was given recently.
    const Mesh* mesh;

    void setMesh(const Mesh* mesh) {
        connection_with_mesh.disconnect();
        this->mesh = mesh;
        if (this->mesh)
            connection_with_mesh = const_cast<Mesh*>(this->mesh)->changed.connect(boost::bind(&SameMeshChecker::onMeshChanged, this, _1), boost::signals2::at_front);
    }

    /**
     * Refresh bounding box cache. Called by mesh changed signal.
     */
    void onMeshChanged(const Mesh::Event&) {
        setMesh(nullptr);
    }

public:

    /**
     * Check if this operator been recently called with given @p mesh and this @p mesh has not been changed since that time.
     * @param mesh mesh to check
     * @return @c true only if this operator been recently called with given @p mesh and this @p mesh has not been changed since that time.
     */
    const bool operator()(const Mesh* mesh) {
        if (this->mesh == mesh)
            return true;
        else {
            setMesh(mesh);
            return false;
        }
    }

    /**
     * Check if this operator been recently called with given @p mesh and this @p mesh has not been changed since that time.
     * @param mesh mesh to check
     * @return @c true only if this operator been recently called with given @p mesh and this @p mesh has not been changed since that time.
     */
    const bool operator()(shared_ptr<const Mesh> mesh) {
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

    ~SameMeshChecker() {
        connection_with_mesh.disconnect();
    }

};


}   // namespace plask

#endif // PLASK__MESH_UTILS_H

